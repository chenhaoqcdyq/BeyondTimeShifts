"""
Evaluate the AV-Sync model on a validation set and report pairwise ranking
accuracy (overall + easy/medium/hard) and score correlation.

Example:
    python evaluate.py \
        --weights ./avsync_eval_weights.pt \
        --data_root /path/to/Crop_5s_resize \
        --batch_size 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
# The bundled qwen2_5_omni / qwen2_vl packages live under avsync_eval/.
sys.path.insert(0, os.path.join(_ROOT, "avsync_eval"))

from avsync_eval.data.dataset import AV_ValDataset
from avsync_eval.metrics import compute_pair_accuracy
from avsync_eval.models.evaluator import AVSyncEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Converted .pt weights (see convert_checkpoint.py)")
    p.add_argument("--data_root", required=True, help="Validation data root")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--pairs_file", default="valing_pairs.json")
    p.add_argument("--scores_file", default="overall_scores.json")
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--v_fps", type=int, default=12)
    p.add_argument("--v_size", type=int, default=140)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision_mode", default="bf16", choices=["bf16", "bf16-mixed"],
                   help="bf16-mixed replicates the training/validate.py autocast path")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="flash_attention_2 matches the AMD-cluster training kernel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="./eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    print("Building model ...")
    model = AVSyncEvaluator(
        model_name=args.model_name,
        v_fps=args.v_fps,
        v_size=args.v_size,
        device=args.device,
        dtype=torch.bfloat16,
        precision_mode=args.precision_mode,
        attn_implementation=args.attn_implementation,
    )
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_eval_checkpoint(state_dict)
    model.to_eval_device()

    # Data
    dataset = AV_ValDataset(
        data_root=args.data_root,
        pairs_file=args.pairs_file,
        scores_file=args.scores_file,
        video_kwargs={"fps": args.v_fps, "shape": args.v_size},
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Inference
    results = []
    for batch in tqdm(loader, desc="Scoring"):
        scores = model.score_batch(batch["videos"], batch["audios"])
        for i in range(len(batch["sample_name"])):
            results.append({
                "sample_name": batch["sample_name"][i],
                "method": batch["method"][i],
                "predicted_score": scores[i],
                "ground_truth_score": batch["score"][i].item(),
            })

    # Metrics
    score_map = {
        (r["sample_name"], r["method"]): {
            "pred_score": r["predicted_score"],
            "gt_score": r["ground_truth_score"],
        }
        for r in results
    }
    with open(os.path.join(args.data_root, args.pairs_file), "r") as f:
        pairs = json.load(f)
    pair_res = compute_pair_accuracy(score_map, pairs)

    gt = np.array([r["ground_truth_score"] for r in results])
    pred = np.array([r["predicted_score"] for r in results])
    correlation = float(np.corrcoef(gt, pred)[0, 1]) if len(gt) > 1 else float("nan")
    mae = float(np.mean(np.abs(gt - pred)))

    print("\n" + "=" * 60)
    print(f"Total pairs   : {pair_res['total_pairs']}")
    print(f"Correct pairs : {pair_res['correct_pairs']}")
    print(f"Pair Accuracy : {pair_res['accuracy']:.4f}")
    for diff, st in pair_res["difficulty_stats"].items():
        if st["total"] > 0:
            print(f"  {diff:7s}: {st['correct']}/{st['total']} = {st['correct']/st['total']:.4f}")
    print(f"Correlation   : {correlation:.4f}")
    print(f"MAE           : {mae:.4f}")
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump(
            {
                "predictions": results,
                "pair_accuracy": pair_res["accuracy"],
                "difficulty_stats": pair_res["difficulty_stats"],
                "correlation": correlation,
                "mae": mae,
            },
            f,
            indent=2,
        )
    print(f"Saved detailed results to {args.out}")


if __name__ == "__main__":
    main()
