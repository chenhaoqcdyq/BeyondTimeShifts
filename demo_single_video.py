"""
Score a single video's audio-video synchronization.

Example:
    python demo_single_video.py \
        --weights ./avsync_eval_weights.pt \
        --video /path/to/some_video.mp4
"""
import argparse
import os
import sys

import numpy as np
import torch
from qwen_omni_utils import process_mm_info

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "avsync_eval"))

from avsync_eval.models.evaluator import AVSyncEvaluator

TARGET_AUDIO = 80000


def load_clip(video_path, fps, size):
    conversation = {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is the video: "},
            {
                "type": "video",
                "video": video_path,
                "fps": fps,
                "resized_height": size,
                "resized_width": size,
            },
        ],
    }
    audios, _, videos = process_mm_info([conversation], use_audio_in_video=True)
    v, a = videos[0], audios[0]
    if isinstance(v, torch.Tensor):
        v = v.numpy()
    if isinstance(a, torch.Tensor):
        a = a.numpy()

    target_v = fps * 5
    if len(v) < target_v:
        pad = np.concatenate([np.zeros((1, 3, size, size)) for _ in range(target_v - len(v))], axis=0)
        v = np.concatenate([v, pad], axis=0)
    elif len(v) > target_v:
        v = v[:target_v]
    if len(a) < TARGET_AUDIO:
        a = np.pad(a, (0, TARGET_AUDIO - len(a)), mode="constant")
    elif len(a) > TARGET_AUDIO:
        a = a[:TARGET_AUDIO]
    return torch.from_numpy(v), torch.from_numpy(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--v_fps", type=int, default=12)
    p.add_argument("--v_size", type=int, default=140)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model = AVSyncEvaluator(
        model_name=args.model_name, v_fps=args.v_fps, v_size=args.v_size, device=args.device
    )
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_eval_checkpoint(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to_eval_device()

    v, a = load_clip(args.video, args.v_fps, args.v_size)
    score = model.score_batch([v], [a])[0]
    print(f"\nSync score for {args.video}: {score:.4f}")


if __name__ == "__main__":
    main()
