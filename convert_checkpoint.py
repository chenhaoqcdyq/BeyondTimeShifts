"""
Convert a DeepSpeed ZeRO checkpoint directory into a single clean state_dict
file suitable for open-source distribution and inference.

The trained checkpoint is a DeepSpeed ZeRO directory:

    best-epoch081-acc0.7323.ckpt/
        checkpoint/
            mp_rank_00_model_states.pt          # <- full fp32/bf16 model weights
            bf16_zero_pp_rank_*_optim_states.pt # <- optimizer shards (NOT needed)
        latest
        zero_to_fp32.py

Only `mp_rank_00_model_states.pt` is needed for inference. Its `module` key holds
the LightningModule state_dict. Since training kept a reference model
(`ref_transformer` / `ref_regression_head`), we drop those and keep only the
`transformer.` and `regression_head.` weights actually used at inference.

Usage:
    python convert_checkpoint.py \
        --ckpt_dir /path/to/best-epoch081-acc0.7323.ckpt \
        --out      ./avsync_eval_weights.pt
"""
import argparse
from pathlib import Path

import torch


def extract_state_dict(ckpt_dir: str):
    ckpt_dir = Path(ckpt_dir)
    model_states = ckpt_dir / "checkpoint" / "mp_rank_00_model_states.pt"
    if not model_states.exists():
        raise FileNotFoundError(f"Cannot find {model_states}")

    print(f"Loading {model_states} ...")
    raw = torch.load(model_states, map_location="cpu")
    if "module" not in raw:
        raise KeyError("Expected a 'module' key in the DeepSpeed model_states file.")
    full_sd = raw["module"]

    # Keep only inference-relevant weights (single policy model + its head).
    kept = {}
    for k, v in full_sd.items():
        if k.startswith("ref_"):
            continue
        if k.startswith("transformer.") or k.startswith("regression_head."):
            kept[k] = v

    print(f"Kept {len(kept)} / {len(full_sd)} tensors "
          f"(dropped reference model + non-inference keys).")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="DeepSpeed ckpt directory")
    parser.add_argument("--out", default="./avsync_eval_weights.pt", help="Output .pt path")
    args = parser.parse_args()

    sd = extract_state_dict(args.ckpt_dir)
    torch.save({"state_dict": sd}, args.out)
    print(f"Saved clean inference weights to: {args.out}")


if __name__ == "__main__":
    main()
