"""
Inference-only Qwen2.5-Omni Thinker wrapper for the AV-Sync evaluator.

Only the multimodal `forward` (with batch-shape normalization for video/audio)
is kept here; all RL/PPO rollout helpers used during training were removed.
"""
import os
import sys
from typing import Optional, List

import torch

# Make the bundled local `qwen2_5_omni` package importable regardless of CWD.
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../avsync_eval
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
)


class hacked_Qwen_Thinker(Qwen2_5OmniThinkerForConditionalGeneration):
    """Qwen2.5-Omni Thinker with batch-preserving multimodal forward."""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ):
        pixel_values_videos, video_grid_thw = self.process_batch_video_input(
            pixel_values_videos, video_grid_thw
        )
        input_features, feature_attention_mask, audio_feature_lengths = self.process_batch_audio_input(
            input_features, feature_attention_mask, audio_feature_lengths
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Merge text / audio / image / video embeddings (prefill stage).
        if input_ids is not None and input_ids.shape[1] != 1:
            if input_features is not None:
                audio_features = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_mask = (
                    (input_ids == self.config.audio_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def process_batch_video_input(self, pixel_values_videos, video_grid_thw):
        """Normalize various video input shapes; for the preprocessed
        2D/3D patch format produced by the Qwen video processor this is a no-op."""
        if pixel_values_videos is None:
            return None, None

        if len(pixel_values_videos.shape) == 4:  # [B, C, H, W] single frame
            batch_size = pixel_values_videos.shape[0]
            if video_grid_thw is None:
                video_grid_thw = torch.tensor(
                    [[1, pixel_values_videos.shape[2], pixel_values_videos.shape[3]]] * batch_size,
                    device=pixel_values_videos.device,
                )
            elif video_grid_thw.shape[0] != batch_size:
                video_grid_thw = video_grid_thw.expand(batch_size, -1)

        elif len(pixel_values_videos.shape) == 5:  # [B, T, C, H, W]
            batch_size = pixel_values_videos.shape[0]
            if video_grid_thw is None:
                frames = pixel_values_videos.shape[1]
                height = pixel_values_videos.shape[3]
                width = pixel_values_videos.shape[4]
                video_grid_thw = torch.tensor(
                    [[frames, height, width]] * batch_size,
                    device=pixel_values_videos.device,
                )
            elif video_grid_thw.shape[0] != batch_size:
                video_grid_thw = video_grid_thw.expand(batch_size, -1)

        # 2D [patches, features] and 3D [patches, temporal, features] are already
        # in the encoder's expected layout -> return unchanged.
        return pixel_values_videos, video_grid_thw

    def process_batch_audio_input(self, input_features, feature_attention_mask, audio_feature_lengths):
        """Pass-through for audio inputs (kept for forward-call signature parity)."""
        if input_features is None:
            return None, None, None
        return input_features, feature_attention_mask, audio_feature_lengths
