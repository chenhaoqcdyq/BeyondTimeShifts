"""
Training datasets for the AV-Sync evaluator.

Three dataset variants, one per training mode:

    AV_Trainset       -> SFT / pairwise-RL. Curriculum-based: each level file
                         lists method pairs of increasing difficulty. Returns
                         two (video, audio) streams per item plus the GT
                         preference indicator.
    AV_RLRankDataset  -> listwise RL_rank. Returns K method streams per video
                         sample for global-ranking (NDCG/Kendall/...) rewards.

Both read videos through `qwen_omni_utils.process_mm_info` (audio is taken from
the video track) and pad/trim every clip to a fixed video-frame / audio-sample
length so items can be batched.

Expected layout under `data_root`:

    data_root/
        overall_scores.json                 # {sample: {method: gt_score}}
        train.txt                           # sample names, one per line (RL_rank)
        curriculumn_SFT/level_{0..9}.json   # curriculum pairs (SFT / pairwise-RL)
        <method>/<sample_name>.mp4          # one mp4 per (method, sample)
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from qwen_omni_utils import process_mm_info


# Generation methods available in the dataset (GT_A is the ground-truth audio).
ALL_METHODS = [
    "foleycontrol", "audiox", "cafa", "foley", "hunyuan",
    "lova", "melqcd", "mmaudio", "selva", "vta_ldm",
]

# A couple of source samples share a video with sample "441"; remap them.
_SAMPLE_REMAP = {"193286_0": "441", "1251": "441"}

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


def _pad_or_trim(videos, audios, dummy_frame, target_frames, target_audio):
    """In-place pad/trim of the video-frame and audio-sample dimensions."""
    for i in range(len(videos)):
        if isinstance(videos[i], torch.Tensor):
            videos[i] = videos[i].numpy()
        if isinstance(audios[i], torch.Tensor):
            audios[i] = audios[i].numpy()

        v_len = len(videos[i])
        if v_len < target_frames:
            pad = np.concatenate([dummy_frame for _ in range(target_frames - v_len)], axis=0)
            videos[i] = np.concatenate([videos[i], pad], axis=0)
        elif v_len > target_frames:
            videos[i] = videos[i][:target_frames]

        a_len = len(audios[i])
        if a_len < target_audio:
            audios[i] = np.pad(audios[i], (0, target_audio - a_len), mode="constant")
        elif a_len > target_audio:
            audios[i] = audios[i][:target_audio]
    return videos, audios


class AV_Trainset(Dataset):
    """Curriculum dataset for SFT / pairwise-RL.

    Each curriculum level file (`curriculumn_{mode}/level_{level}.json`) maps a
    sample name to a list of pairs. Every item yields two (video, audio) streams
    (method_1, method_2, order fixed) and an `indicator` (+1 if method_1 is the
    better method by GT score, -1 otherwise).

    `distributed_cnt` sets the per-level sample budget for one epoch (the level
    files are usually larger than what a single epoch consumes).
    """

    def __init__(
        self,
        data_root,
        level,
        distributed_cnt,
        training_mode="SFT",
        video_kwargs=None,
        sample_lst=None,
    ):
        self.data_root = data_root
        self.level = level
        self.distributed_cnt = distributed_cnt
        self.video_kwargs = video_kwargs or {"fps": 12, "shape": 140}

        with open(f"{data_root}/overall_scores.json", "r") as f:
            self.score_dict = json.load(f)

        if sample_lst is None:
            with open(f"{data_root}/curriculumn_{training_mode}/level_{level}.json", "r") as f:
                pair_dict = json.load(f)
            self.sample_lst = []
            for sample in pair_dict.keys():
                for pair in pair_dict[sample]:
                    self.sample_lst.append([sample, pair])
        else:
            self.sample_lst = sample_lst

        self.dummy_video_frame = np.zeros((1, 3, self.video_kwargs["shape"], self.video_kwargs["shape"]))
        self.target_video_num_frame = self.video_kwargs["fps"] * 5
        self.target_audio_num_frame = 80000
        np.random.shuffle(self.sample_lst)
        print(f"[AV_Trainset] level {level}: {len(self.sample_lst)} pairs, "
              f"epoch budget {distributed_cnt}")

    def __len__(self):
        return self.distributed_cnt

    def _build_conversation(self, video_path):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the video: "},
                {
                    "type": "video",
                    "video": video_path,
                    "fps": self.video_kwargs["fps"],
                    "resized_height": self.video_kwargs["shape"],
                    "resized_width": self.video_kwargs["shape"],
                },
            ],
        }

    def __getitem__(self, index):
        sample_name, pair = self.sample_lst[index % len(self.sample_lst)]
        sample_name = _SAMPLE_REMAP.get(sample_name, sample_name)
        method_1, method_2 = pair[0], pair[1]

        score_1 = self.score_dict[sample_name][method_1]
        score_2 = self.score_dict[sample_name][method_2]
        indicator = 1 if score_1 > score_2 else -1

        v1 = os.path.join(self.data_root, method_1, sample_name + ".mp4")
        v2 = os.path.join(self.data_root, method_2, sample_name + ".mp4")
        conversations = [self._build_conversation(v1), self._build_conversation(v2)]
        audios, _, videos = process_mm_info(conversations, use_audio_in_video=True)

        videos, audios = _pad_or_trim(
            videos, audios, self.dummy_video_frame,
            self.target_video_num_frame, self.target_audio_num_frame,
        )
        stacked_videos = np.stack(videos, axis=0)   # (2, frames, 3, H, W)
        stacked_audios = np.stack(audios, axis=0)    # (2, audio_frames)

        return {
            "sample_name": sample_name,
            "method_1": method_1,
            "method_2": method_2,
            "audios": torch.from_numpy(stacked_audios),
            "videos": torch.from_numpy(stacked_videos),
            "indicator": indicator,
            "gt_scores": torch.tensor([score_1, score_2], dtype=torch.float32),
        }


class AV_RLRankDataset(Dataset):
    """Listwise dataset for RL_rank: K methods per video sample.

    Each item returns K method streams for one video, enabling global ranking
    rewards (NDCG, Kendall Tau, Spearman, Top-1, ...). GT_A is always included
    when `include_gt` and present for the sample.

    Args:
        data_root:        Root data directory.
        sample_list_path: txt file of sample names (one per line). If None, uses
                          all samples in overall_scores.json.
        num_methods:      Methods to load per sample. <=0 or > available loads
                          all; otherwise randomly samples this many.
        video_kwargs:     {'fps', 'shape'} for video processing.
        include_gt:       Whether to include GT_A in the candidate list.
    """

    def __init__(
        self,
        data_root,
        sample_list_path=None,
        num_methods=6,
        video_kwargs=None,
        include_gt=True,
    ):
        self.data_root = data_root
        self.video_kwargs = video_kwargs or {"fps": 12, "shape": 140}
        self.include_gt = include_gt

        self.all_methods = list(ALL_METHODS)
        if include_gt:
            self.all_methods.append("GT_A")

        self.num_methods = num_methods
        if self.num_methods <= 0 or self.num_methods > len(self.all_methods):
            self.num_methods = len(self.all_methods)

        with open(f"{data_root}/overall_scores.json", "r") as f:
            self.score_dict = json.load(f)

        if sample_list_path is not None:
            with open(sample_list_path, "r") as f:
                self.sample_names = [line.strip() for line in f if line.strip()]
        else:
            self.sample_names = list(self.score_dict.keys())
        self.sample_names = [s for s in self.sample_names if s not in _SAMPLE_REMAP]

        self.dummy_video_frame = np.zeros((1, 3, self.video_kwargs["shape"], self.video_kwargs["shape"]))
        self.target_video_num_frame = self.video_kwargs["fps"] * 5
        self.target_audio_num_frame = 80000
        print(f"[AV_RLRankDataset] {len(self.sample_names)} samples, "
              f"{self.num_methods} methods/sample (include_gt={include_gt})")

    def __len__(self):
        return len(self.sample_names)

    def _select_methods(self, sample_name):
        available = [m for m in self.all_methods if m in self.score_dict.get(sample_name, {})]
        if len(available) <= self.num_methods:
            return available
        if self.include_gt and "GT_A" in available:
            non_gt = [m for m in available if m != "GT_A"]
            selected = list(np.random.choice(non_gt, self.num_methods - 1, replace=False))
            selected.append("GT_A")
        else:
            selected = list(np.random.choice(available, self.num_methods, replace=False))
        return selected

    def __getitem__(self, index):
        sample_name = self.sample_names[index % len(self.sample_names)]
        sample_name = _SAMPLE_REMAP.get(sample_name, sample_name)

        methods = self._select_methods(sample_name)
        gt_scores = np.array([self.score_dict[sample_name][m] for m in methods], dtype=np.float32)
        gt_rank_order = np.argsort(-gt_scores).copy()

        conversations = []
        for method in methods:
            video_path = os.path.join(self.data_root, method, sample_name + ".mp4")
            conversations.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is the video: "},
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.video_kwargs["fps"],
                        "resized_height": self.video_kwargs["shape"],
                        "resized_width": self.video_kwargs["shape"],
                    },
                ],
            })

        audios, _, videos = process_mm_info(conversations, use_audio_in_video=True)
        videos, audios = _pad_or_trim(
            videos, audios, self.dummy_video_frame,
            self.target_video_num_frame, self.target_audio_num_frame,
        )
        stacked_videos = np.stack(videos, axis=0)   # (K, frames, 3, H, W)
        stacked_audios = np.stack(audios, axis=0)    # (K, audio_frames)

        return {
            "sample_name": sample_name,
            "methods": methods,
            "num_methods": len(methods),
            "audios": torch.from_numpy(stacked_audios),
            "videos": torch.from_numpy(stacked_videos),
            "gt_scores": torch.from_numpy(gt_scores),
            "gt_rank_order": torch.from_numpy(gt_rank_order),
        }


def rl_rank_collate_fn(batch):
    """Collate for AV_RLRankDataset: pad all items to the batch's max K.

    For batch_size=1 (the recommended setting) this is effectively a no-op.
    """
    max_K = max(item["num_methods"] for item in batch)

    padded_videos, padded_audios = [], []
    padded_gt_scores, padded_gt_rank_order = [], []
    for item in batch:
        K = item["num_methods"]
        pad = max_K - K
        if pad > 0:
            v_pad = torch.zeros(pad, *item["videos"].shape[1:], dtype=item["videos"].dtype)
            a_pad = torch.zeros(pad, *item["audios"].shape[1:], dtype=item["audios"].dtype)
            s_pad = torch.zeros(pad, dtype=item["gt_scores"].dtype)
            r_pad = torch.zeros(pad, dtype=item["gt_rank_order"].dtype)
            padded_videos.append(torch.cat([item["videos"], v_pad], dim=0))
            padded_audios.append(torch.cat([item["audios"], a_pad], dim=0))
            padded_gt_scores.append(torch.cat([item["gt_scores"], s_pad], dim=0))
            padded_gt_rank_order.append(torch.cat([item["gt_rank_order"], r_pad], dim=0))
        else:
            padded_videos.append(item["videos"])
            padded_audios.append(item["audios"])
            padded_gt_scores.append(item["gt_scores"])
            padded_gt_rank_order.append(item["gt_rank_order"])

    return {
        "sample_name": [item["sample_name"] for item in batch],
        "methods": [item["methods"] for item in batch],
        "num_methods": torch.tensor([item["num_methods"] for item in batch]),
        "videos": torch.stack(padded_videos, dim=0),
        "audios": torch.stack(padded_audios, dim=0),
        "gt_scores": torch.stack(padded_gt_scores, dim=0),
        "gt_rank_order": torch.stack(padded_gt_rank_order, dim=0),
    }
