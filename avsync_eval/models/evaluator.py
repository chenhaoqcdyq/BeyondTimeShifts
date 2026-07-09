"""
AVSyncEvaluator: minimal single-model inference wrapper.

Given a video (audio is read from the video track), it produces a scalar
audio-video synchronization score. This is the inference-only counterpart of
the training-time `MyPLModule`, stripped of the reference model, rollout, and
RL machinery.

Scoring pipeline:
    video+audio  ->  Qwen2.5-Omni Thinker  ->  last-layer last-token hidden
                 ->  Linear regression head  ->  scalar sync score
"""
import os

import numpy as np
import torch
import torch.nn as nn

from qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from .hacked_qwen import hacked_Qwen_Thinker

# Token id of the assistant role marker ("<|im_start|>assistant") in the Qwen
# tokenizer; the score is read from the position right after this token.
# (Kept for reference; the final-token readout below does not require it.)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"

SYS_PROMPT = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You will be given a video and an audio, and you need "
            "to rate the synchronization between the video and the audio.",
        }
    ],
}
ASSISTANT_RESPONSE = {"role": "assistant", "content": [{"type": "text", "text": "SCORE"}]}


class MLP_RegressionHead(nn.Module):
    """Single linear layer mapping the LLM hidden state to a scalar score.

    The name `fc_1` and the (input_dim -> output_dim) wiring match the trained
    checkpoint exactly; `hidden_dim` is accepted but unused.
    """

    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=1):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc_1(x)


class AVSyncEvaluator(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        v_fps: int = 12,
        v_size: int = 140,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        precision_mode: str = "bf16",
        attn_implementation: str = "sdpa",
    ):
        """
        precision_mode:
            "bf16"        -> pure bf16 weights, no autocast (fastest).
            "bf16-mixed"  -> bf16 weights + torch.autocast on forward. This
                             replicates the training/validate.py path
                             (Lightning Trainer precision="bf16-mixed" on a
                             model that was also .to(bfloat16)), where reduction
                             ops accumulate in fp32. Use this to reproduce the
                             exact training-time scores.
        attn_implementation:
            "sdpa"             -> PyTorch scaled_dot_product_attention (default).
            "flash_attention_2"-> FlashAttention-2 kernel; matches the AMD-cluster
                             training run (which used use_flash_attention_2=True).
                             Requires the `flash-attn` package to be installed.
            "eager"            -> plain math attention.
        """
        super().__init__()
        self.v_fps = v_fps
        self.v_size = v_size
        self.device_str = device
        self.dtype = dtype
        assert precision_mode in ("bf16", "bf16-mixed"), precision_mode
        self.precision_mode = precision_mode

        # FlashAttention-2 requires a bf16/fp16 model and a CUDA device, so for
        # FA2 we load directly in the target dtype on the target device. For
        # sdpa/eager we keep the validate.py path (load fp32 on CPU, cast later).
        if attn_implementation == "flash_attention_2":
            load_kwargs = dict(torch_dtype=self.dtype, device_map=self.device_str)
        else:
            load_kwargs = dict(torch_dtype=torch.float32, device_map="cpu")
        self.transformer = hacked_Qwen_Thinker.from_pretrained(
            model_name, attn_implementation=attn_implementation, **load_kwargs
        )
        self.regression_head = MLP_RegressionHead(2048, 1024, 1)
        self.preprocessor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def load_eval_checkpoint(self, state_dict):
        """Load a flat state_dict (keys prefixed with `transformer.` and
        `regression_head.`) as produced by `convert_checkpoint.py`."""
        transformer_ckpt = {
            k.replace("transformer.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("transformer.")
        }
        head_ckpt = {
            k.replace("regression_head.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("regression_head.")
        }

        missing, unexpected = self.transformer.load_state_dict(transformer_ckpt, strict=False)
        print(f"[transformer] missing={len(missing)} unexpected={len(unexpected)}")
        missing, unexpected = self.regression_head.load_state_dict(head_ckpt, strict=False)
        print(f"[regression_head] missing={len(missing)} unexpected={len(unexpected)}")

    def to_eval_device(self):
        """Move to target device, cast to inference dtype, set eval mode.

        The checkpoint was trained with DeepSpeed bf16; casting to bf16 here
        reproduces the online training numerics (see validate.py notes).

        Idempotent w.r.t. dtype/device: for FA2 the transformer is already loaded
        in bf16 on CUDA, and casting again is a no-op; the regression head still
        needs to be moved/cast here."""
        self.to(self.device_str)
        self.to(self.dtype)
        self.eval()
        return self

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    def _build_conversation(self, video_tensor):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the video: "},
                {
                    "type": "video",
                    "video": video_tensor,
                    "fps": self.v_fps,
                    "resized_height": self.v_size,
                    "resized_width": self.v_size,
                },
            ],
        }

    def _prepare_inputs(self, video_list, audio_list):
        conversations = [
            [SYS_PROMPT, self._build_conversation(v), ASSISTANT_RESPONSE] for v in video_list
        ]
        text = self.preprocessor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False, use_audio_in_video=True
        )
        inputs = self.preprocessor(
            text=text,
            audio=audio_list,
            images=None,
            videos=video_list,
            return_tensors="pt",
            padding=True,
            videos_kwargs={
                "insert_timestamps": True,
                "do_resize": False,
                "use_audio_in_video": True,
                "fps": self.v_fps,
            },
        )
        inputs = inputs.to(self.device_str)
        # Drop the trailing token so the model predicts at the "SCORE" position,
        # matching the training-time readout.
        inputs["input_ids"] = inputs["input_ids"][:, :-1]
        inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
        # Cast float inputs (e.g. audio features) to the model dtype.
        for key in inputs.keys():
            if hasattr(inputs[key], "dtype") and inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(self.dtype)
        return inputs

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_batch(self, videos, audios):
        """
        Args:
            videos: tensor (B, frames, 3, H, W) or list of such tensors
            audios: tensor (B, audio_len) or list / ndarray
        Returns:
            list[float] of length B with sync scores
        """
        video_list, audio_list = [], []
        for i in range(len(videos)):
            v = videos[i]
            a = audios[i]
            if isinstance(a, torch.Tensor):
                a = a.float().cpu().numpy()
            else:
                a = np.asarray(a, dtype=np.float32)
            video_list.append(v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            audio_list.append(a)

        inputs = self._prepare_inputs(video_list, audio_list)
        if self.precision_mode == "bf16-mixed":
            # Replicate Lightning Trainer precision="bf16-mixed": autocast lets
            # reduction-heavy ops accumulate in fp32 even with bf16 weights.
            dev_type = "cuda" if str(self.device_str).startswith("cuda") else "cpu"
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
                outputs = self.transformer(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1][:, -1]
                scores = self.regression_head(last_hidden)
        else:
            outputs = self.transformer(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1]  # (B, hidden)
            scores = self.regression_head(last_hidden)       # (B, 1)
        return scores.squeeze(1).float().cpu().tolist()
