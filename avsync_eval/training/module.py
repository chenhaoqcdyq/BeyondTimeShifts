"""
AVSyncTrainModule: the training-time LightningModule for the AV-Sync evaluator.

This is the counterpart of the inference-only `AVSyncEvaluator`. It wraps a
policy Qwen2.5-Omni Thinker + linear score head, plus a frozen reference copy
used by the RL objectives, and supports three training modes:

    SFT      - supervised cold start. Cross-entropy on the "SCORE" token plus a
               Bradley-Terry pairwise ranking loss on the predicted scores.
    RL       - pairwise GRPO. Gaussian rollouts around the predicted score;
               reward = correct pairwise ordering / global ranking of the pair.
    RL_rank  - listwise GRPO. K methods per sample; reward = global ranking
               quality (NDCG / Kendall / Spearman / Top-1 / MRR / pairwise).

The mode is chosen per batch: RL_rank when the batch carries `num_methods`
(from AV_RLRankDataset); otherwise `train_mode` selects SFT vs pairwise RL.

Model / head layout matches `convert_checkpoint.py` and `AVSyncEvaluator`
exactly, so a checkpoint trained here can be converted and evaluated with the
existing inference package.
"""
import math
import os
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from avsync_eval.models.hacked_qwen import hacked_Qwen_Thinker
from avsync_eval.metrics import compute_pair_accuracy
from avsync_eval.training.ranking_reward import (
    compute_ranking_reward,
    compute_ranking_reward_for_rollouts,
)

# Token id of the Qwen assistant-role marker ("<|im_start|>assistant").
ASSISTANT_TOKEN_ID = 77091

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
SCORE_RESPONSE = {"role": "assistant", "content": [{"type": "text", "text": "SCORE"}]}


class MLP_RegressionHead(nn.Module):
    """Single linear layer (input_dim -> output_dim). `hidden_dim` is accepted
    for signature parity with the checkpoint but unused, matching the trained
    weights (`fc_1`)."""

    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=1):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc_1(x)


class AVSyncTrainModule(L.LightningModule):
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-Omni-3B",
        transformer=None,          # optional path to a .pt checkpoint to init from
        train_mode="RL_rank",      # 'SFT' | 'RL' | 'RL_rank' (RL_rank auto-detected per batch)
        v_fps=12,
        v_size=140,
        # RL / GRPO hyper-parameters
        epison=0.2,
        KL_weight=1e-3,
        manual_std=2.5,
        num_rollout=12,
        # optimizer / schedule
        learning_rate=1e-6,
        weight_decay=5e-2,
        wp=1,
        wp0=0.005,
        wpe=0.1,
        twde=0,
        check_window=200,
        curriculum_threshold=0.88,  # advance to a harder lesson above this epoch acc (SFT/RL)
        # validation
        pairs_path=None,           # valing_pairs.json for val pair-accuracy
        val_output_dir="./validation_results",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transformer"])

        self.train_mode = train_mode
        self.v_fps = v_fps
        self.v_size = v_size
        self.epison = epison
        self.KL_weight = KL_weight
        self.manual_std = manual_std
        self.num_rollout = num_rollout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        self.twde = twde or weight_decay
        self.check_window = check_window
        self.curriculum_threshold = curriculum_threshold
        self.lession = 0  # current curriculum lesson; synced from datamodule at train start
        self.pairs_path = pairs_path
        self.val_output_dir = val_output_dir

        self.sys_prompt_dict = SYS_PROMPT

        # Policy model + frozen reference model (reference used by RL objectives).
        self.transformer = hacked_Qwen_Thinker.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        self.ref_transformer = hacked_Qwen_Thinker.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        self.regression_head = MLP_RegressionHead(2048, 1024, 1)
        self.ref_regression_head = MLP_RegressionHead(2048, 1024, 1)

        if transformer is not None:
            self.load_checkpoint(transformer)

        # Freeze reference model + encoders + embeddings; train only LLM layers + head.
        self.transformer.train()
        self.ref_transformer.eval()
        for p in self.ref_transformer.parameters():
            p.requires_grad = False
        for name, param in self.transformer.named_parameters():
            if "audio_tower" in name or "visual" in name:
                param.requires_grad = False
            elif "model" in name and "embed_tokens" in name:
                param.requires_grad = False
            else:
                param.requires_grad = "layers" in name
        for p in self.regression_head.parameters():
            p.requires_grad = True
        for p in self.ref_regression_head.parameters():
            p.requires_grad = False
        self.regression_head.train()

        self.preprocessor = Qwen2_5OmniProcessor.from_pretrained(model_name)

        self.loss_history = deque(maxlen=self.check_window)
        self.validation_step_outputs = []
        self._validation_output_dir = None

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def load_checkpoint(self, ckpt_path):
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint {ckpt_path} not found, skipping load.")
            return
        print(f"Loading checkpoint from {ckpt_path} ...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

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
        for model in (self.transformer, self.ref_transformer):
            model.load_state_dict(transformer_ckpt, strict=False)
        for head in (self.regression_head, self.ref_regression_head):
            head.load_state_dict(head_ckpt, strict=False)
        print("Checkpoint loaded.")

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, input_dict, only_logits=True):
        outputs = self.transformer(**input_dict, output_hidden_states=True)
        pred_scores = self.regression_head(outputs.hidden_states[-1][:, -1])
        return (pred_scores, outputs.logits) if only_logits else (pred_scores, outputs)

    @torch.no_grad()
    def forward_ref(self, input_dict, only_logits=True):
        outputs = self.ref_transformer(**input_dict, output_hidden_states=True)
        pred_scores = self.ref_regression_head(outputs.hidden_states[-1][:, -1])
        return (pred_scores, outputs.logits) if only_logits else (pred_scores, outputs)

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    def _build_conversation(self, video):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the video: "},
                {
                    "type": "video",
                    "video": video,
                    "fps": self.v_fps,
                    "resized_height": self.v_size,
                    "resized_width": self.v_size,
                },
            ],
        }

    def prepare_qwen_inputs(self, conversation, v_lst, a_lst):
        text = self.preprocessor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, use_audio_in_video=True
        )
        inputs = self.preprocessor(
            text=text, audio=a_lst, images=None, videos=v_lst,
            return_tensors="pt", padding=True,
            videos_kwargs={
                "insert_timestamps": True,
                "do_resize": False,
                "use_audio_in_video": True,
                "fps": self.v_fps,
            },
        )
        return inputs.to(self.device)

    def _teacher_force_and_mask(self, original_seq_dict, original_dtype):
        """Shift inputs by one, build labels + loss mask over the assistant span."""
        inputs = deepcopy(original_seq_dict)
        inputs["input_ids"] = inputs["input_ids"][:, :-1]
        inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
        labels = original_seq_dict["input_ids"][:, 1:]
        assistant_pos = torch.where(inputs["input_ids"] == ASSISTANT_TOKEN_ID)

        loss_mask = torch.zeros_like(inputs["input_ids"])
        for i in range(inputs["input_ids"].size(0)):
            batch_mask = assistant_pos[0] == i
            if batch_mask.any():
                start = assistant_pos[1][batch_mask][0]
                idx = torch.arange(inputs["input_ids"].size(1), device=inputs["input_ids"].device)
                loss_mask[i, idx >= start] = 1
            else:
                print(f"Warning: no assistant token in sample {i}, masking all positions.")
                loss_mask[i, :] = 1

        for key in inputs.keys():
            if hasattr(inputs[key], "dtype") and inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(original_dtype)
        return inputs, labels, loss_mask

    def get_xc(self, batch, with_label=True):
        """Pairwise batch (SFT / RL): stack method_1 then method_2 along dim 0."""
        videos, audios = batch["videos"].detach().cpu(), batch["audios"].detach().cpu()
        original_dtype = audios.dtype
        method_1_v, method_2_v = videos[:, 0], videos[:, 1]
        method_1_a, method_2_a = audios[:, 0], audios[:, 1]
        indicator = batch["indicator"]

        conv_1, conv_2, v1, v2, a1, a2 = [], [], [], [], [], []
        for n in range(len(method_1_v)):
            v1.append(method_1_v[n]); v2.append(method_2_v[n])
            a1.append(method_1_a[n].float().numpy()); a2.append(method_2_a[n].float().numpy())
            tail = [SCORE_RESPONSE] if with_label else []
            conv_1.append([self.sys_prompt_dict, self._build_conversation(method_1_v[n])] + tail)
            conv_2.append([self.sys_prompt_dict, self._build_conversation(method_2_v[n])] + tail)

        conversations = conv_1 + conv_2
        video_lst = v1 + v2
        audio_lst = a1 + a2
        original_seq_dict = self.prepare_qwen_inputs(conversations, video_lst, audio_lst)

        if not with_label:
            return original_seq_dict, indicator
        inputs, labels, loss_mask = self._teacher_force_and_mask(original_seq_dict, original_dtype)
        return inputs, labels, loss_mask, indicator

    def get_xc_val(self, batch, with_label=True):
        """Validation batch: one (video, audio) per item."""
        videos, audios = batch["videos"].detach().cpu(), batch["audios"].detach().cpu()
        original_dtype = next(self.parameters()).dtype

        conversations, video_lst, audio_lst = [], [], []
        for n in range(len(videos)):
            video_lst.append(videos[n])
            audio_lst.append(audios[n].float().numpy())
            tail = [SCORE_RESPONSE] if with_label else []
            conversations.append([self.sys_prompt_dict, self._build_conversation(videos[n])] + tail)

        original_seq_dict = self.prepare_qwen_inputs(conversations, video_lst, audio_lst)
        if not with_label:
            return original_seq_dict
        inputs, labels, loss_mask = self._teacher_force_and_mask(original_seq_dict, original_dtype)
        return inputs, labels, loss_mask

    def get_xc_rank(self, batch, with_label=True):
        """Listwise batch (RL_rank): flatten (B, K) valid methods to (total, ...)."""
        videos = batch["videos"].detach().cpu()
        audios = batch["audios"].detach().cpu()
        gt_scores = batch["gt_scores"]
        num_methods = batch["num_methods"]
        methods = batch["methods"]
        original_dtype = audios.dtype
        B = videos.shape[0]

        conversations, video_lst, audio_lst = [], [], []
        for b_idx in range(B):
            for k_idx in range(num_methods[b_idx].item()):
                video_lst.append(videos[b_idx, k_idx])
                audio_lst.append(audios[b_idx, k_idx].float().numpy())
                tail = [SCORE_RESPONSE] if with_label else []
                conversations.append([self.sys_prompt_dict, self._build_conversation(videos[b_idx, k_idx])] + tail)

        original_seq_dict = self.prepare_qwen_inputs(conversations, video_lst, audio_lst)
        if not with_label:
            return original_seq_dict, gt_scores, num_methods, methods
        inputs, labels, loss_mask = self._teacher_force_and_mask(original_seq_dict, original_dtype)
        return inputs, labels, loss_mask, gt_scores, num_methods, methods

    # ------------------------------------------------------------------ #
    # Rollouts / rewards
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rollout(self, pred_scores):
        """Gaussian rollout around each predicted score -> (N, num_rollout)."""
        pred_scores = pred_scores.squeeze(1)
        sampled = []
        for i in range(pred_scores.size(0)):
            delta = torch.normal(mean=0.0, std=self.manual_std, size=(self.num_rollout,)).to(pred_scores.device)
            sampled.append(delta + pred_scores[i])
        return torch.stack(sampled, dim=0)

    def compute_reward(self, pred_scores, indicator):
        """Pairwise 0/1 reward per rollout (reference logging for pairwise RL)."""
        num_batch = len(pred_scores)
        rewards = []
        for i in range(num_batch // 2):
            pre, post = pred_scores[i].squeeze(), pred_scores[i + num_batch // 2].squeeze()
            pred_ind = torch.where(pre > post, 1, -1)
            rewards.append((pred_ind == indicator[i]).float())
        return torch.stack(rewards).repeat(2, 1)

    # ------------------------------------------------------------------ #
    # Training steps
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        if "num_methods" in batch:
            loss = self.shared_step_RL_ranking(batch, batch_idx)
            bs = batch["gt_scores"].shape[0]
        elif self.train_mode == "SFT":
            loss = self.shared_step_SFT(batch, batch_idx)
            bs = self.trainer.datamodule.batch_size
        else:
            loss = self.shared_step_RL(batch, batch_idx)
            bs = self.trainer.datamodule.batch_size
        self.log("train/loss", loss, sync_dist=True, prog_bar=False, logger=True, batch_size=bs)
        return loss

    def shared_step_SFT(self, batch, batch_idx, mode="train"):
        input_dict, label_ids, loss_mask, indicator = self.get_xc(batch)
        loss_mask = loss_mask.reshape(-1)

        pred_scores, logits = self(input_dict)
        pre_scores, post_scores = pred_scores.split(pred_scores.size(0) // 2, dim=0)
        pred_indicator = torch.where(pre_scores > post_scores, 1, -1).squeeze(1)
        batch_acc = (pred_indicator == indicator).float().mean()
        self.loss_history.append(batch_acc)

        bt_loss = -F.logsigmoid(indicator.unsqueeze(1) * (pre_scores - post_scores)).mean()

        temp_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), label_ids.reshape(-1), reduction="none"
        )
        mask_sum = loss_mask.sum()
        if mask_sum > 0:
            ce_loss = (loss_mask * temp_loss).sum() / mask_sum
        else:
            print(f"Warning: empty loss mask at batch {batch_idx}.")
            ce_loss = temp_loss.sum() * 0.0

        bs = self.trainer.datamodule.batch_size
        self.log(f"{mode}/ce_loss", ce_loss, sync_dist=True, prog_bar=False, logger=True, batch_size=bs)
        self.log(f"{mode}/bt_loss", bt_loss, sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        self.log(f"{mode}/acc", batch_acc, sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        return ce_loss + bt_loss

    def shared_step_RL(self, batch, batch_idx, mode="train"):
        input_dict, _, _, indicator = self.get_xc(batch)

        pred_scores, logits = self(input_dict)
        ref_scores, _ = self.forward_ref(input_dict)
        KL_term = (self.manual_std ** 2 + torch.square(pred_scores - ref_scores)) / (2 * self.manual_std ** 2) - 0.5

        B = pred_scores.size(0) // 2
        pre_scores, post_scores = pred_scores.split(B, dim=0)
        pred_indicator = torch.where(pre_scores.squeeze(1) > post_scores.squeeze(1), 1, -1)
        batch_acc = (pred_indicator == indicator.squeeze()).float()

        gt_scores_pair = batch["gt_scores"].to(pred_scores.device)
        pred_scores_pair = torch.cat([pre_scores, post_scores], dim=1)
        current_rank = compute_ranking_reward(pred_scores_pair.detach(), gt_scores_pair.float())

        rollout_scores = self.rollout(pred_scores)
        pre_rollout, post_rollout = rollout_scores.split(B, dim=0)
        rollout_scores_pair = torch.stack([pre_rollout, post_rollout], dim=1)  # (B, 2, num_rollout)
        ranking_rewards = compute_ranking_reward_for_rollouts(rollout_scores_pair, gt_scores_pair.float())

        reward_mean = ranking_rewards.mean(dim=1, keepdim=True)
        reward_std = ranking_rewards.std(dim=1, keepdim=True)
        advantage = ((ranking_rewards - reward_mean) / (reward_std + 1e-8)).repeat(2, 1)

        mu_theta = pred_scores.repeat(1, self.num_rollout)
        mu_ref = ref_scores.detach().repeat(1, self.num_rollout)
        ratio = torch.exp(((rollout_scores - mu_ref) ** 2 - (rollout_scores - mu_theta) ** 2) / (2 * self.manual_std ** 2))
        item_clip = torch.clamp(ratio, 1 - self.epison, 1 + self.epison) * advantage
        item_orig = ratio * advantage
        grpo_loss = -torch.mean(torch.min(item_clip, item_orig))

        bs = self.trainer.datamodule.batch_size
        self.loss_history.append(batch_acc.mean())
        self.log("RL/loss", grpo_loss, sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("RL/KL", KL_term.mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        self.log(f"{mode}/acc", batch_acc.mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("rank/top1", current_rank["top1"].mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("rewards/rank_mean", reward_mean.mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=bs)
        return grpo_loss + self.KL_weight * KL_term.mean()

    def shared_step_RL_ranking(self, batch, batch_idx, mode="train"):
        inputs, labels, loss_mask, gt_scores, num_methods, methods = self.get_xc_rank(batch)
        B, K = gt_scores.shape
        total_items = sum(num_methods[b].item() for b in range(B))

        pred_scores_flat, logits = self(inputs)
        ref_scores_flat, _ = self.forward_ref(inputs)
        KL_term = (self.manual_std ** 2 + torch.square(pred_scores_flat - ref_scores_flat)) / (2 * self.manual_std ** 2) - 0.5

        pred_grouped = torch.zeros(B, K, device=pred_scores_flat.device, dtype=pred_scores_flat.dtype)
        validity_mask = torch.zeros(B, K, device=pred_scores_flat.device)
        offset = 0
        for b_idx in range(B):
            kv = num_methods[b_idx].item()
            pred_grouped[b_idx, :kv] = pred_scores_flat[offset:offset + kv].squeeze(1)
            validity_mask[b_idx, :kv] = 1.0
            offset += kv

        gt_scores_device = gt_scores.to(pred_scores_flat.device).float()
        current_rank = compute_ranking_reward(pred_grouped.detach(), gt_scores_device, mask=validity_mask)

        pair_acc_list = []
        for b_idx in range(B):
            kv = num_methods[b_idx].item()
            ps, gs = pred_grouped[b_idx, :kv], gt_scores_device[b_idx, :kv]
            correct, total_p = 0, 0
            for i in range(kv):
                for j in range(i + 1, kv):
                    if gs[i] == gs[j]:
                        continue
                    total_p += 1
                    if (ps[i] > ps[j]) == (gs[i] > gs[j]):
                        correct += 1
            pair_acc_list.append(correct / total_p if total_p > 0 else 0.5)
        batch_pair_acc = torch.tensor(pair_acc_list, device=pred_scores_flat.device).mean()

        rollout_flat = self.rollout(pred_scores_flat)  # (total_items, num_rollout)
        rollout_grouped = torch.zeros(B, K, self.num_rollout, device=rollout_flat.device, dtype=rollout_flat.dtype)
        offset = 0
        for b_idx in range(B):
            kv = num_methods[b_idx].item()
            rollout_grouped[b_idx, :kv, :] = rollout_flat[offset:offset + kv]
            offset += kv

        rollout_rewards = compute_ranking_reward_for_rollouts(rollout_grouped, gt_scores_device, mask=validity_mask)
        reward_mean = rollout_rewards.mean(dim=1, keepdim=True)
        reward_std = rollout_rewards.std(dim=1, keepdim=True)
        advantage = (rollout_rewards - reward_mean) / (reward_std + 1e-8)

        advantage_expanded = torch.zeros(total_items, self.num_rollout, device=advantage.device, dtype=advantage.dtype)
        offset = 0
        for b_idx in range(B):
            kv = num_methods[b_idx].item()
            for k_idx in range(kv):
                advantage_expanded[offset + k_idx] = advantage[b_idx]
            offset += kv

        mu_theta = pred_scores_flat.repeat(1, self.num_rollout)
        mu_ref = ref_scores_flat.detach().repeat(1, self.num_rollout)
        ratio = torch.exp(((rollout_flat - mu_ref) ** 2 - (rollout_flat - mu_theta) ** 2) / (2 * self.manual_std ** 2))
        item_clip = torch.clamp(ratio, 1 - self.epison, 1 + self.epison) * advantage_expanded
        item_orig = ratio * advantage_expanded
        grpo_loss = -torch.mean(torch.min(item_clip, item_orig))

        self.loss_history.append(batch_pair_acc.item())
        self.log("RL_rank/loss", grpo_loss, sync_dist=True, prog_bar=True, logger=True, batch_size=B)
        self.log("RL_rank/KL", KL_term.mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=B)
        self.log("RL_rank/pair_acc", batch_pair_acc, sync_dist=True, prog_bar=True, logger=True, batch_size=B)
        self.log("RL_rank/ndcg", current_rank["ndcg"].mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=B)
        self.log("RL_rank/top1", current_rank["top1"].mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=B)
        self.log("RL_rank/reward_mean", reward_mean.mean(), sync_dist=True, prog_bar=True, logger=True, batch_size=B)
        return grpo_loss + self.KL_weight * KL_term.mean()

    # ------------------------------------------------------------------ #
    # Optimizer / schedule
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        param_dict.update({f"regression_head.{pn}": p for pn, p in self.regression_head.named_parameters()})
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

    # ------------------------------------------------------------------ #
    # Epoch hooks
    # ------------------------------------------------------------------ #
    def on_train_start(self):
        # Sync the curriculum lesson from the datamodule (may be resumed / set via CLI).
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "num_lession"):
            self.lession = dm.num_lession

    def on_train_epoch_end(self):
        if len(self.loss_history) == 0:
            return
        loss_tensor = torch.tensor(list(self.loss_history), device=self.device)
        if self.trainer.world_size > 1:
            gathered = [torch.zeros_like(loss_tensor) for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather(gathered, loss_tensor)
            mean_acc = torch.cat(gathered).mean().item()
        else:
            mean_acc = float(np.mean(list(self.loss_history)))
        if self.global_rank == 0:
            print(f"Epoch {self.current_epoch} mean acc: {mean_acc:.4f}")
        self.loss_history.clear()

        # Dynamic curriculum: advance to a harder lesson once the running
        # accuracy clears the threshold (SFT / pairwise RL only; RL_rank has no
        # curriculum). The datamodule reloads its dataloader each epoch, so the
        # new lesson's difficulty mix takes effect next epoch.
        dm = getattr(self.trainer, "datamodule", None)
        if self.train_mode == "RL_rank" or dm is None or not hasattr(dm, "num_lession"):
            self.log("lession", float(self.lession), sync_dist=True, prog_bar=False, logger=True)
            return

        max_lession = getattr(dm, "max_lession", 9)
        if mean_acc > self.curriculum_threshold and self.lession < max_lession:
            self.lession = min(self.lession + 1, max_lession)
            if self.global_rank == 0:
                print(f"[curriculum] acc {mean_acc:.4f} > {self.curriculum_threshold} "
                      f"-> advancing to lesson {self.lession}")
        elif self.global_rank == 0:
            print(f"[curriculum] acc {mean_acc:.4f} <= {self.curriculum_threshold} "
                  f"-> staying at lesson {self.lession}")
        dm.num_lession = self.lession
        self.log("lession", float(self.lession), sync_dist=True, prog_bar=False, logger=True)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        input_dict = self.get_xc_val(batch, with_label=True)[0]
        pred_scores, _ = self(input_dict)
        for i in range(len(batch["sample_name"])):
            self.validation_step_outputs.append({
                "sample_name": batch["sample_name"][i],
                "method": batch["method"][i],
                "predicted_score": pred_scores[i].item(),
                "ground_truth_score": batch["score"][i].item() if "score" in batch else None,
            })
        return {}

    def on_validation_epoch_end(self):
        import json
        from pathlib import Path

        if self.trainer.world_size > 1:
            gathered = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(gathered, self.validation_step_outputs)
            all_results = [r for rank_results in gathered for r in rank_results]
        else:
            all_results = self.validation_step_outputs

        pair_accuracy = torch.tensor(0.0, device=self.device)
        acc_easy = torch.tensor(0.0, device=self.device)
        acc_medium = torch.tensor(0.0, device=self.device)
        acc_hard = torch.tensor(0.0, device=self.device)

        if self.global_rank == 0 and len(all_results) > 0:
            score_map = {
                (r["sample_name"], r["method"]): {
                    "pred_score": r["predicted_score"],
                    "gt_score": r["ground_truth_score"],
                }
                for r in all_results
            }
            if self.pairs_path and os.path.exists(self.pairs_path):
                with open(self.pairs_path, "r") as f:
                    pairs = json.load(f)
                pair_res = compute_pair_accuracy(score_map, pairs)
                pair_accuracy = torch.tensor(pair_res["accuracy"], device=self.device)
                for diff, st in pair_res["difficulty_stats"].items():
                    if st["total"] > 0:
                        acc = st["correct"] / st["total"]
                        if diff == "easy":
                            acc_easy = torch.tensor(acc, device=self.device)
                        elif diff == "medium":
                            acc_medium = torch.tensor(acc, device=self.device)
                        elif diff == "hard":
                            acc_hard = torch.tensor(acc, device=self.device)
            else:
                print(f"Warning: pairs file {self.pairs_path} not set/found; pair accuracy = 0.")

            out_dir = Path(self.val_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"validation_results_epoch_{self.current_epoch}.json", "w") as f:
                json.dump(all_results, f, indent=2)

            if all_results[0].get("ground_truth_score") is not None:
                gt = [r["ground_truth_score"] for r in all_results]
                pred = [r["predicted_score"] for r in all_results]
                print(f"Correlation: {np.corrcoef(gt, pred)[0, 1]:.4f}  "
                      f"MAE: {np.mean(np.abs(np.array(gt) - np.array(pred))):.4f}")

        if self.trainer.world_size > 1:
            for t in (pair_accuracy, acc_easy, acc_medium, acc_hard):
                torch.distributed.broadcast(t, src=0)

        self.log("val/pair_accuracy", pair_accuracy.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=False)
        self.log("val/acc_easy", acc_easy.item(), on_epoch=True, prog_bar=False, logger=True, sync_dist=False, rank_zero_only=False)
        self.log("val/acc_medium", acc_medium.item(), on_epoch=True, prog_bar=False, logger=True, sync_dist=False, rank_zero_only=False)
        self.log("val/acc_hard", acc_hard.item(), on_epoch=True, prog_bar=False, logger=True, sync_dist=False, rank_zero_only=False)
        self.validation_step_outputs.clear()
