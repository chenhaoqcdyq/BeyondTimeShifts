"""
Batch-score generative-model video outputs for audio-video synchronization.

Clips have NO ground-truth labels, so we produce per-clip sync scores and
per-model aggregate statistics (mean/median) — useful for ranking generation
models against each other. Clips without an audio track are skipped and reported.

Two directory layouts are supported via --layout:

  flat   (default): one sub-directory per model, each holding clips directly.
      root/
        modelA/001.mp4  002.mp4  ...
        modelB/001.mp4  002.mp4  ...

  nested: one sub-directory per sample, each holding several candidate clips
      (e.g. the LTX set with multiple seeds/takes per prompt). Each sample is
      reduced to a single representative score via --nested_reduce.
      root/
        1/video_00.mp4  video_00_seed1234.mp4  video_01.mp4 ...
        2/...

Examples:
    # Flat: rank several models' outputs
    python tools/score_videogen.py --weights ./avsync_eval_weights.pt \
        --root /path/to/VideoGen/outputs --out ./videogen_scores.json

    # Nested: score an LTX-style set (many candidates per sample)
    python tools/score_videogen.py --weights ./avsync_eval_weights.pt \
        --root /path/to/LTX --layout nested --model_label LTX --out ./ltx_scores.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm

# This script lives in tools/; the project root is one level up.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "avsync_eval"))

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
    p.add_argument("--root", required=True, help="Root dir (see --layout)")
    p.add_argument("--layout", default="flat", choices=["flat", "nested"],
                   help="flat: per-model dirs of clips; nested: per-sample dirs of candidates")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--models", nargs="*", default=None,
                   help="[flat] sub-dir names to score; default = all with mp4s")
    p.add_argument("--model_label", default="LTX",
                   help="[nested] name used for the single model in the summary")
    p.add_argument("--nested_reduce", default="median_high",
                   choices=["median_high", "median_low", "mean", "best", "first"],
                   help="[nested] how to reduce each sample's candidates to one score")
    p.add_argument("--v_fps", type=int, default=12)
    p.add_argument("--v_size", type=int, default=140)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--limit", type=int, default=0, help="Max clips per model (0=all, flat only)")
    p.add_argument("--out", default="./videogen_scores.json")
    return p.parse_args()


def build_model(args):
    model = AVSyncEvaluator(
        model_name=args.model_name, v_fps=args.v_fps, v_size=args.v_size,
        device=args.device, dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_eval_checkpoint(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to_eval_device()
    return model


def score_one(model, path, args):
    v, a = load_clip(path, args.v_fps, args.v_size)
    return float(model.score_batch([v], [a])[0])


# ---------------------------------------------------------------- flat --------
def discover_models(root, requested):
    if requested:
        return requested
    models = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and any(f.endswith(".mp4") for f in os.listdir(d)):
            models.append(name)
    return models


def run_flat(model, args):
    models = discover_models(args.root, args.models)
    print(f"[flat] Scoring models: {models}")
    results, skipped = {}, {}
    for m in models:
        mdir = os.path.join(args.root, m)
        clips = sorted(f for f in os.listdir(mdir) if f.endswith(".mp4"))
        if args.limit > 0:
            clips = clips[: args.limit]
        results[m], skipped[m] = {}, []
        for clip in tqdm(clips, desc=m):
            path = os.path.join(mdir, clip)
            if not has_audio(path):
                skipped[m].append(clip)
                continue
            try:
                results[m][clip] = score_one(model, path, args)
            except Exception as e:
                skipped[m].append(f"{clip} (error: {type(e).__name__})")
    return results, skipped


# -------------------------------------------------------------- nested --------
def _reduce(cand_scores, how):
    """cand_scores: {clip_name: score}. Returns (rep_clip, rep_score)."""
    items = sorted(cand_scores.items(), key=lambda kv: kv[1])  # ascending
    n = len(items)
    if how == "first":
        # first candidate in natural filename order
        first = sorted(cand_scores.items())[0]
        return first
    if how == "best":
        return items[-1]
    if how == "mean":
        return ("<mean>", float(np.mean([s for _, s in items])))
    if how == "median_low":
        return items[(n - 1) // 2]
    # median_high (default)
    return items[n // 2]


def run_nested(model, args):
    sids = sorted(
        (d for d in os.listdir(args.root)
         if os.path.isdir(os.path.join(args.root, d))),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"[nested] {len(sids)} samples, reduce={args.nested_reduce}, label={args.model_label}")
    per_candidate = {}   # sid -> {clip_key: {"ours_score": s}}   (full, for diffing)
    reduced = {}         # sid -> representative score
    skipped = []
    for sid in tqdm(sids, desc=args.model_label):
        sdir = os.path.join(args.root, sid)
        clips = sorted(f for f in os.listdir(sdir) if f.endswith(".mp4"))
        cand = {}
        for clip in clips:
            path = os.path.join(sdir, clip)
            if not has_audio(path):
                skipped.append(f"{sid}/{clip} (no audio)")
                continue
            try:
                cand[clip[:-4]] = score_one(model, path, args)
            except Exception as e:
                skipped.append(f"{sid}/{clip} ({type(e).__name__})")
        if not cand:
            continue
        per_candidate[sid] = {k: {"ours_score": v} for k, v in cand.items()}
        rep_clip, rep_score = _reduce(cand, args.nested_reduce)
        reduced[sid] = {"rep_clip": rep_clip, "ours_score": rep_score}
    return per_candidate, reduced, skipped


# ---------------------------------------------------------------- main --------
def summarize(results, skipped):
    print("\n" + "=" * 68)
    print(f"{'model':<26}{'n':>5}{'mean':>10}{'median':>10}{'std':>10}{'skip':>7}")
    print("-" * 68)
    summary = {}
    for m in results:
        scores = np.array(list(results[m].values()), dtype=np.float64)
        sk = len(skipped.get(m, []))
        if len(scores) == 0:
            print(f"{m:<26}{0:>5}{'-':>10}{'-':>10}{'-':>10}{sk:>7}")
            summary[m] = {"n": 0, "skipped": sk}
            continue
        summary[m] = {"n": int(len(scores)), "mean": float(scores.mean()),
                      "median": float(np.median(scores)), "std": float(scores.std()), "skipped": sk}
        print(f"{m:<26}{len(scores):>5}{scores.mean():>10.3f}"
              f"{np.median(scores):>10.3f}{scores.std():>10.3f}{sk:>7}")
    print("=" * 68)
    ranked = sorted((m for m in results if summary[m].get("n", 0) > 0),
                    key=lambda m: summary[m]["mean"], reverse=True)
    print("\nRanking by mean sync score (higher = better):")
    for i, m in enumerate(ranked, 1):
        print(f"  {i}. {m:<24} mean={summary[m]['mean']:.3f}  (n={summary[m]['n']})")
    return summary


def main():
    args = parse_args()
    model = build_model(args)

    if args.layout == "flat":
        results, skipped = run_flat(model, args)
        summary = summarize(results, skipped)
        payload = {"summary": summary, "per_clip": results, "skipped": skipped}
    else:
        per_candidate, reduced, skipped = run_nested(model, args)
        # reduced representative scores -> single-model summary
        rep = {args.model_label: {sid: v["ours_score"] for sid, v in reduced.items()}}
        summary = summarize(rep, {args.model_label: skipped})
        payload = {
            "summary": summary,
            "reduce": args.nested_reduce,
            "reduced": reduced,           # sid -> {rep_clip, ours_score}
            "per_candidate": per_candidate,  # sid -> {clip: {ours_score}}  (full)
            "skipped": skipped,
        }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
