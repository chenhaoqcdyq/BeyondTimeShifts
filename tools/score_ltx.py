"""
Score the LTX closed-video set, which has a NESTED layout:
    LTX/<sample_id>/video_XX.mp4  and video_XX_seed1234.mp4

Produces the same JSON shape as the reference ltx_ours_scores.json:
    { "<sample_id>": { "<video_key>": {"ours_score": <float>}, ... }, ... }
so it can be diffed clip-by-clip.
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

# This script lives in tools/; the project root is one level up.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "avsync_eval"))

from avsync_eval.models.evaluator import AVSyncEvaluator
from demo_single_video import load_clip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--root", required=True, help="LTX dir with <sid>/video_*.mp4")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--v_fps", type=int, default=12)
    p.add_argument("--v_size", type=int, default=140)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--out", default="./ltx_scores_full.json")
    return p.parse_args()


def main():
    args = parse_args()
    model = AVSyncEvaluator(
        model_name=args.model_name, v_fps=args.v_fps, v_size=args.v_size,
        device=args.device, dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_eval_checkpoint(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to_eval_device()

    sids = sorted(
        (d for d in os.listdir(args.root)
         if os.path.isdir(os.path.join(args.root, d))),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    results = {}
    skipped = []
    total = 0
    for sid in tqdm(sids, desc="LTX"):
        sdir = os.path.join(args.root, sid)
        clips = sorted(f for f in os.listdir(sdir) if f.endswith(".mp4"))
        results[sid] = {}
        for clip in clips:
            key = clip[:-4]  # strip .mp4
            path = os.path.join(sdir, clip)
            try:
                v, a = load_clip(path, args.v_fps, args.v_size)
                score = model.score_batch([v], [a])[0]
                results[sid][key] = {"ours_score": float(score)}
                total += 1
            except Exception as e:
                skipped.append(f"{sid}/{clip} ({type(e).__name__})")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScored {total} clips across {len(sids)} samples. Skipped {len(skipped)}.")
    if skipped:
        print("Skipped:", skipped[:10])
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
