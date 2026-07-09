"""
Train the AV-Sync evaluator (Qwen2.5-Omni Thinker + linear score head).

Three modes (see avsync_eval/training/module.py for the objectives):

    SFT      supervised cold start   (curriculum pairs, Bradley-Terry + CE)
    RL       pairwise GRPO           (curriculum pairs)
    RL_rank  listwise GRPO           (K methods per sample, ranking reward)

The released pipeline is a two-stage recipe: SFT cold start, then RL_rank from
the SFT checkpoint. Example:

    # Stage 1 - SFT cold start
    python train.py --train_mode SFT \
        --data_root /path/to/Crop_5s_resize \
        --exp_dir ./runs/sft --devices 6

    # Stage 2 - listwise RL from the SFT checkpoint
    #   (convert the SFT DeepSpeed ckpt with convert_checkpoint.py first)
    python train.py --train_mode RL_rank \
        --data_root /path/to/Crop_5s_resize \
        --pretrained ./runs/sft_weights.pt \
        --sample_list_path /path/to/Crop_5s_resize/train.txt \
        --exp_dir ./runs/rl_rank --devices 6

Data layout is documented in README.md (Training section) and
avsync_eval/training/train_dataset.py.
"""
import argparse
import os
import sys

import lightning as L
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "avsync_eval"))

from avsync_eval.data.dataset import AV_ValDataset
from avsync_eval.training.module import AVSyncTrainModule
from avsync_eval.training.train_dataset import (
    AV_RLRankDataset,
    AV_Trainset,
    rl_rank_collate_fn,
)

# Curriculum schedule for SFT / pairwise-RL. Each row is a "lesson"; the 10
# columns are the sampling weights for difficulty levels level_0 .. level_9
# (level_0 = hardest / smallest GT-score gap, level_9 = easiest). Training starts
# at row `--num_lession` and advances one row each time the running accuracy
# clears `--curriculum_threshold`, shifting mass from easy toward hard levels.
DISTRIBUTE_LST = [
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.7500],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0500, 0.3000, 0.6500],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0500, 0.1000, 0.3500, 0.5000],
    [0.0355, 0.0355, 0.0355, 0.0355, 0.0355, 0.0355, 0.2130, 0.2130, 0.1923, 0.1686],
    [0.0417, 0.0417, 0.0417, 0.0417, 0.0417, 0.0417, 0.2292, 0.2014, 0.1736, 0.1458],
    [0.0516, 0.0516, 0.0516, 0.0516, 0.0570, 0.0656, 0.2194, 0.1849, 0.1505, 0.1161],
    [0.0571, 0.0571, 0.0738, 0.1071, 0.0810, 0.0905, 0.1714, 0.1333, 0.1143, 0.1143],
    [0.0609, 0.0901, 0.1193, 0.1484, 0.0865, 0.0948, 0.1000, 0.1000, 0.1000, 0.1000],
    [0.0934, 0.1170, 0.1406, 0.1641, 0.0808, 0.0808, 0.0808, 0.0808, 0.0808, 0.0808],
    [0.1200, 0.1405, 0.1585, 0.1585, 0.0704, 0.0704, 0.0704, 0.0704, 0.0704, 0.0704],
    [0.1500, 0.1500, 0.1500, 0.1500, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667],
]
MAX_LESSION = len(DISTRIBUTE_LST) - 1
NUMSAMPLE_PER_EPOCH = 6000


class AVSyncDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.video_kwargs = {"fps": args.v_fps, "shape": args.v_size}
        # Mutable curriculum state: the module bumps this after each epoch and
        # train_dataloader (reloaded every epoch) reads the latest value.
        self.num_lession = args.num_lession
        self.max_lession = MAX_LESSION

    def setup(self, stage=None):
        self.val_dataset = AV_ValDataset(
            data_root=self.args.data_root,
            pairs_file=self.args.pairs_file,
            scores_file=self.args.scores_file,
            video_kwargs=self.video_kwargs,
        )

    def train_dataloader(self):
        args = self.args
        if args.train_mode == "RL_rank":
            self.train_dataset = AV_RLRankDataset(
                data_root=args.data_root,
                sample_list_path=args.sample_list_path,
                num_methods=args.num_methods,
                video_kwargs=self.video_kwargs,
                include_gt=args.include_gt,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=rl_rank_collate_fn,
            )
        # SFT / pairwise RL: curriculum ConcatDataset over the 10 levels.
        # Read the (possibly advanced) lesson index, not the static CLI default.
        counts = [int(NUMSAMPLE_PER_EPOCH * x) for x in DISTRIBUTE_LST[self.num_lession]]
        self.train_dataset = ConcatDataset([
            AV_Trainset(
                data_root=args.data_root,
                level=i,
                distributed_cnt=counts[i],
                training_mode=args.train_mode,
                video_kwargs=self.video_kwargs,
            )
            for i in range(10)
        ])
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_root", required=True)
    p.add_argument("--sample_list_path", default=None, help="train.txt for RL_rank (optional)")
    p.add_argument("--pairs_file", default="valing_pairs.json")
    p.add_argument("--scores_file", default="overall_scores.json")
    # mode / model
    p.add_argument("--train_mode", default="RL_rank", choices=["SFT", "RL", "RL_rank"])
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--pretrained", default=None, help="Init from a converted .pt checkpoint")
    p.add_argument("--v_fps", type=int, default=12)
    p.add_argument("--v_size", type=int, default=140)
    # RL_rank data
    p.add_argument("--num_methods", type=int, default=6)
    p.add_argument("--include_gt", action="store_true", default=True)
    p.add_argument("--no_include_gt", dest="include_gt", action="store_false")
    p.add_argument("--num_lession", type=int, default=0,
                   help="Starting curriculum lesson row (SFT/RL)")
    p.add_argument("--curriculum_threshold", type=float, default=0.88,
                   help="Advance to the next (harder) lesson when epoch accuracy exceeds this (SFT/RL)")
    # GRPO / optim
    p.add_argument("--epison", type=float, default=0.2)
    p.add_argument("--kl_weight", type=float, default=1e-3)
    p.add_argument("--manual_std", type=float, default=2.5)
    p.add_argument("--num_rollout", type=int, default=12)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--check_window", type=int, default=200)
    # batching
    p.add_argument("--batch_size", type=int, default=None,
                   help="Default: 1 for RL_rank, 8 for SFT/RL")
    p.add_argument("--val_batch_size", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    # trainer
    p.add_argument("--devices", type=int, default=6)
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--strategy", default="deepspeed_stage_2")
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_dir", default="./runs/avsync")
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_size is None:
        args.batch_size = 1 if args.train_mode == "RL_rank" else 8

    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    module = AVSyncTrainModule(
        model_name=args.model_name,
        transformer=args.pretrained,
        train_mode=args.train_mode,
        v_fps=args.v_fps,
        v_size=args.v_size,
        epison=args.epison,
        KL_weight=args.kl_weight,
        manual_std=args.manual_std,
        num_rollout=args.num_rollout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        check_window=args.check_window,
        curriculum_threshold=args.curriculum_threshold,
        pairs_path=os.path.join(args.data_root, args.pairs_file),
        val_output_dir=os.path.join(args.exp_dir, "validation_results"),
    )
    datamodule = AVSyncDataModule(args)

    logger = TensorBoardLogger(save_dir=os.path.join(args.exp_dir, "tb"), name="", version="v1")
    best_ckpt = ModelCheckpoint(
        dirpath=args.exp_dir,
        filename="best-epoch{epoch:03d}-acc{val/pair_accuracy:.4f}",
        monitor="val/pair_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator="gpu",
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=5,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
        callbacks=[lr_monitor, best_ckpt],
        logger=logger,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
