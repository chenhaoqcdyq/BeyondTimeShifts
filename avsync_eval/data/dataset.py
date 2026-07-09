"""
Validation dataset for AV-Sync evaluation.

Each item is a single (sample, method) video. The audio track is read from the
video file via `process_mm_info`. Videos/audios are padded or trimmed to a fixed
length so they can be batched.

Expected directory layout under `data_root`:
    data_root/
        overall_scores.json        # {sample_name: {method: gt_score, ...}, ...}
        valing_pairs.json          # {sample_name: [[method1, method2, diff], ...]}
        <method>/<sample_name>.mp4  # one mp4 per (method, sample)
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from qwen_omni_utils import process_mm_info


class AV_ValDataset(Dataset):
    def __init__(
        self,
        data_root,
        pairs_file="valing_pairs.json",
        scores_file="overall_scores.json",
        video_kwargs=None,
    ):
        self.data_root = data_root
        self.video_kwargs = video_kwargs or {"fps": 12, "shape": 140}

        with open(os.path.join(data_root, scores_file), "r") as f:
            self.score_dict = json.load(f)
        with open(os.path.join(data_root, pairs_file), "r") as f:
            pair_dict = json.load(f)

        # Build the list of unique (sample, method) pairs referenced by the pairs file.
        self.sample_lst = []
        seen = set()
        for sample in pair_dict.keys():
            for pair in pair_dict[sample]:
                for method in (pair[0], pair[1]):
                    key = (sample, method)
                    if key not in seen:
                        seen.add(key)
                        self.sample_lst.append((sample, method))

        self.dummy_video_frame = np.zeros((1, 3, self.video_kwargs["shape"], self.video_kwargs["shape"]))
        self.target_video_num_frame = self.video_kwargs["fps"] * 5
        self.target_audio_num_frame = 80000
        print(f"[AV_ValDataset] {len(self.sample_lst)} unique (sample, method) items.")

    def __len__(self):
        return len(self.sample_lst)

    def __getitem__(self, index):
        sampled_name, method = self.sample_lst[index]
        score = self.score_dict[sampled_name][method]
        video_path = os.path.join(self.data_root, method, sampled_name + ".mp4")

        conversation = {
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
        audios, _, videos = process_mm_info([conversation], use_audio_in_video=True)

        if isinstance(videos[0], torch.Tensor):
            videos[0] = videos[0].numpy()
        if isinstance(audios[0], torch.Tensor):
            audios[0] = audios[0].numpy()

        v_len, a_len = len(videos[0]), len(audios[0])
        if v_len < self.target_video_num_frame:
            pad = np.concatenate(
                [self.dummy_video_frame for _ in range(self.target_video_num_frame - v_len)], axis=0
            )
            videos[0] = np.concatenate([videos[0], pad], axis=0)
        elif v_len > self.target_video_num_frame:
            videos[0] = videos[0][: self.target_video_num_frame]

        if a_len < self.target_audio_num_frame:
            audios[0] = np.pad(audios[0], (0, self.target_audio_num_frame - a_len), mode="constant")
        elif a_len > self.target_audio_num_frame:
            audios[0] = audios[0][: self.target_audio_num_frame]

        return {
            "sample_name": sampled_name,
            "method": method,
            "audios": torch.from_numpy(audios[0]),
            "videos": torch.from_numpy(videos[0]),
            "score": float(score),
        }
