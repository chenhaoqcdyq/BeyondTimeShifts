"""
Batch-score VideoGen model outputs for audio-video synchronization.

These clips have NO ground-truth labels, so we only produce per-clip sync
scores and per-model aggregate statistics (mean/median) — useful for ranking
the generation models against each other. Clips without an audio track are
skipped and reported.

Example:
    python score_videogen.py \
        --weights ./avsync_eval_weights.pt \
        --root /cpfs01/qyj_workspace/jcwang/Move2AMDCluster/VideoGen/outputs \
        --out ./videogen_scores.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from avsync_eval.models.evaluator import AVSyncEvaluator
from demo_single_video import load_clip


def has_audio(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return "audio" in out.stdout
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--root", required=True, help="Dir containing per-model subdirs of mp4s")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subdir names to score; default = all subdirs with mp4s")
    p.add_argument("--v_fps", type=int, default=12)
    p.add_argument("--v_size", type=int, default=140)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--limit", type=int, default=0, help="Max clips per model (0=all)")
    p.add_argument("--out", default="./videogen_scores.json")
    return p.parse_args()


def discover_models(root, requested):
    if requested:
        return requested
    models = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and any(f.endswith(".mp4") for f in os.listdir(d)):
            models.append(name)
    return models


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

    models = discover_models(args.root, args.models)
    print(f"Scoring models: {models}")

    results = {}     # model -> {clip: score}
    skipped = {}     # model -> [clips without audio / failed]

    for m in models:
        mdir = os.path.join(args.root, m)
        clips = sorted(f for f in os.listdir(mdir) if f.endswith(".mp4"))
        if args.limit > 0:
            clips = clips[: args.limit]
        results[m] = {}
        skipped[m] = []
        for clip in tqdm(clips, desc=m):
            path = os.path.join(mdir, clip)
            if not has_audio(path):
                skipped[m].append(clip)
                continue
            try:
                v, a = load_clip(path, args.v_fps, args.v_size)
                score = model.score_batch([v], [a])[0]
                results[m][clip] = score
            except Exception as e:
                skipped[m].append(f"{clip} (error: {type(e).__name__})")

    # Aggregate
    print("\n" + "=" * 68)
    print(f"{'model':<26}{'n':>5}{'mean':>10}{'median':>10}{'std':>10}{'skip':>7}")
    print("-" * 68)
    summary = {}
    for m in models:
        scores = np.array(list(results[m].values()), dtype=np.float64)
        if len(scores) == 0:
            print(f"{m:<26}{0:>5}{'-':>10}{'-':>10}{'-':>10}{len(skipped[m]):>7}")
            summary[m] = {"n": 0, "skipped": len(skipped[m])}
            continue
        summary[m] = {
            "n": int(len(scores)),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
            "skipped": len(skipped[m]),
        }
        print(f"{m:<26}{len(scores):>5}{scores.mean():>10.3f}"
              f"{np.median(scores):>10.3f}{scores.std():>10.3f}{len(skipped[m]):>7}")
    print("=" * 68)

    # Rank models by mean sync score (higher = better sync)
    ranked = sorted(
        [m for m in models if summary[m].get("n", 0) > 0],
        key=lambda m: summary[m]["mean"], reverse=True,
    )
    print("\nRanking by mean sync score (higher = better):")
    for i, m in enumerate(ranked, 1):
        print(f"  {i}. {m:<24} mean={summary[m]['mean']:.3f}  (n={summary[m]['n']})")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_clip": results, "skipped": skipped}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
