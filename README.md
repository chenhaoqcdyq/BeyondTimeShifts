<div align="center">

<h1>AV-Sync Evaluator</h1>

<p><strong>🎬 Beyond Time Shifts: A Reference-Free Evaluator for Generative Audio-Visual Models</strong></p>

<p>A reference-free audio-visual synchronization metric built on Qwen2.5-Omni-3B. Give it a single video clip — audio read from the track — and it returns one scalar: how causally consistent the audio and the visuals are. Higher is better, no reference audio required. Official code for our ECCV 2026 paper, with the full training recipe (preference SFT + R-GRPO) and inference stack in one repository.</p>

<p>
  <a href="#-quick-start"><b>🚀 Quick Start</b></a> |
  <a href="#-method"><b>🧠 Method</b></a> |
  <a href="#-evaluation"><b>📊 Evaluation</b></a> |
  <a href="#-training"><b>🔧 Training</b></a> |
  <a href="#-use-your-own-dataset"><b>📁 Custom Data</b></a> |
  <a href="#-roadmap"><b>🧭 Roadmap</b></a> |
  <a href="#-citation"><b>📝 Citation</b></a>
</p>

<p>
  <img src="https://img.shields.io/badge/ECCV-2026-1f6feb?style=flat-square" alt="ECCV 2026">
  <img src="https://img.shields.io/badge/Backbone-Qwen2.5--Omni--3B-blue?style=flat-square" alt="Qwen2.5-Omni-3B">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/transformers-4.57.1-yellow?style=flat-square&logo=huggingface&logoColor=white" alt="transformers 4.57.1">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT">
</p>

<img src="assets/overview.png" width="100%" alt="Failure cases and pipeline overview">

<p><sub><b>Top:</b> generative audio-visual outputs exhibit structural hallucinations and asymmetric cross-modal relations that break the "just a time shift" assumption of classic sync metrics. <b>Bottom:</b> our pipeline — the <b>SynthSync</b> dataset of authentic generative failures, a reference-free continuous Omni-LLM scorer, and <b>R-GRPO</b> reinforcement post-training.</sub></p>

</div>

## 🔥 News

- **[2026-07]** 🎉 Full open-source release — inference/evaluation package and the complete two-stage training pipeline (preference SFT + R-GRPO).
- **[2026-07]** 📄 Our paper is accepted to **ECCV 2026**.
- **[coming soon]** 📦 The **SynthSync** dataset and trained checkpoint are being prepared for release. Until then, everything runs on your [own dataset](#-use-your-own-dataset).

## ✨ Introduction

Audio-visual generative models (Sora-2, Veo-3, Seedance) are becoming "world
simulators," but their cross-modal synchronization stays fragile — and, crucially,
their failures are **not** simple temporal offsets. They involve *structural
hallucinations* (a glass-shatter sound when a plastic cup hits carpet),
*asymmetric relations* (a visible action with no corresponding sound), and
*temporally diffuse* events. Classic sync metrics (onset accuracy, contrastive
similarity, offset detection) assume structural correctness and therefore fail on
generated content.

This creates a paradox: **humans judge synchronization by relative comparison**
("which of these two audios fits the video better?"), yet a **deployable metric
must be reference-free and absolute** (one continuous score for one clip). This
repository resolves that paradox with a three-part recipe from the paper:

1. **SynthSync** — the first dataset of *authentic* generative sync failures:
   10 state-of-the-art V2A models generate competing audios for the *same* video,
   and expert annotators provide pairwise preferences (≈306K annotations).
2. **Continuous Omni-LLM evaluator** — a Qwen2.5-Omni-3B whose discrete language
   head is replaced by a continuous projection head reading the hidden state at a
   `[SCORE]` token, yielding a single reference-free scalar.
3. **R-GRPO** (Real-Valued Group Relative Policy Optimization) — reinforcement
   post-training that treats the scalar output as a Gaussian policy and optimizes
   the whole *listwise* score distribution against the ground-truth ranking, so
   the metric internalizes global causal structure rather than local pairwise
   margins.

The result is a metric with state-of-the-art alignment to human preference, which
we use to build **SyncBench**, a standardized leaderboard for AV-generation models.

> This repo is self-contained: it bundles a local copy of the Qwen2.5-Omni /
> Qwen2-VL model code, so only `transformers` is needed for tokenizers/config.

### 🌟 Key Features

|     | Feature | Description |
| --- | ------- | ----------- |
| 🎯 | **Reference-free scoring** | One continuous causal-synchronization score for a single `(video, audio)` clip — no reference audio needed. |
| 🧠 | **Continuous Omni-LLM** | Qwen2.5-Omni-3B with a `[SCORE]`-token projection head, trained by ranking rather than regression. |
| 🔧 | **Full training recipe** | Preference SFT (dynamic curriculum) + R-GRPO reinforcement post-training in one Lightning module; multi-GPU via DeepSpeed ZeRO-2. |
| 📊 | **Batteries-included eval** | Ranking accuracy (NDCG, Kendall τ, Pair-Acc, Top-1), easy/medium/hard breakdown, correlation, MAE — plus batch scorers for whole model outputs. |
| 🔁 | **Reproducible** | Inference replicates bf16 training numerics; training and evaluation share the exact same model + head wiring. |

## 📊 Results

On the SynthSync benchmark, R-GRPO sets a new state of the art, outperforming
off-the-shelf Omni-LLMs, non-LLM metrics, and trained LLM evaluators:

| Metric | PFT baseline | **+ R-GRPO (ours)** |
|--------|:---:|:---:|
| NDCG            | 0.9353 | **0.9435** |
| Kendall's τ     | 0.4515 | **0.4899** |
| MRR             | 0.7316 | **0.7674** |
| Pairwise Acc (%) | 71.16 | **72.38** |

**Component ablation** (NDCG / Pairwise-Acc): base 0.8531 / 51.02 → +SynthSync
0.9224 / 66.60 → +preference SFT 0.9353 / 71.16 → **+R-GRPO 0.9435 / 72.38**.

<sub>Defaults: Qwen2.5-Omni-3B, inputs 5 s @ 12 FPS, `140×140`, listwise K=6, 12 Gaussian rollouts (σ²=2.5), AdamW lr 1e-6, PPO clip ε=0.2, KL β=0.001, bf16.</sub>

## 🚀 Quick Start

```bash
# Install (inference / evaluation)
pip install -r requirements.txt

# 1. Convert the trained DeepSpeed checkpoint into a single clean .pt (once)
python convert_checkpoint.py \
    --ckpt_dir /path/to/best-epoch081-acc0.7323.ckpt \
    --out      ./avsync_eval_weights.pt

# 2. Score a single clip
python demo_single_video.py \
    --weights ./avsync_eval_weights.pt \
    --video   /path/to/clip.mp4
# -> Sync score for /path/to/clip.mp4: 3.14
```

Key versions: `transformers==4.57.1`, `qwen-omni-utils==0.0.8`. The base weights
`Qwen/Qwen2.5-Omni-3B` download automatically from the Hugging Face Hub on first
run. For training, also `pip install -r requirements_train.txt` (adds lightning,
deepspeed, scipy, tensorboard).

### Convert the checkpoint

The trained checkpoint is a DeepSpeed ZeRO directory. `convert_checkpoint.py`
reads only `checkpoint/mp_rank_00_model_states.pt`, keeps the `transformer.*` and
`regression_head.*` tensors, and drops the optimizer shards and the training-time
reference model — so `zero_to_fp32.py` is **not** required.

## 🧠 Method

<div align="center">
<img src="assets/framework.png" width="95%" alt="Framework overview">
</div>

The evaluator is a Qwen2.5-Omni-3B backbone with its discrete language head
replaced by a continuous projection MLP. A persistent `[SCORE]` token is appended
to the multimodal context, and its final hidden state `h_score ∈ ℝ^d` is mapped to
a scalar `s = MLP(h_score)`, stripping away language-generation stochasticity.

Training aligns this scalar to human preference topology in two stages:

- **Stage I — Latent preference alignment (SFT).** Because ground truth is
  *relative*, we skip MSE regression and optimize a Bradley-Terry-Luce objective
  over pairwise preferences `a_i ≻_v a_j`. Fitting all margins through a single
  shared `f_θ` induces a globally consistent, reference-free value space. A
  **dynamic curriculum** starts with easy, high-margin pairs (rank-1 vs rank-9)
  and anneals toward finer comparisons as accuracy improves.
- **Stage II — R-GRPO.** Pairwise learning suffers *metric myopia*: local margins
  don't guarantee a globally transitive scale. R-GRPO reparameterizes the
  deterministic scalar `μ_θ` as a Gaussian policy `N(μ_θ, σ²)`, samples N listwise
  rollouts per anchor video, and rewards each rollout by how well its full ranking
  of the K candidates matches the ground-truth topology (a composite of NDCG,
  Kendall τ, Spearman ρ, Top-1, MRR, pairwise concordance). Group-relative
  advantages, a closed-form Gaussian importance ratio, PPO clipping, and a KL
  penalty to the reference policy complete the objective.

<div align="center">
<img src="assets/rl_curves.png" width="95%" alt="R-GRPO validation learning curves">
<br><sub>Validation learning curves during R-GRPO post-training.</sub>
</div>

### 🏆 SyncBench leaderboard

We deploy the metric to rank six cutting-edge AV-generation models (Sora-2,
Veo-3.1, WAN-2.6, Vidu-Q3, Grok-3, LTX-2) on **185 diverse prompts** spanning a
wide range of physical events. Unlike legacy metrics (AV-Align, JavisScore,
DeSync) — which show extreme variance and rankings that contradict human consensus
— our metric cleanly stratifies model quality.

<div align="center">
<img src="assets/syncbench.png" width="95%" alt="SyncBench content distribution">
<br><sub>SyncBench content distribution: prompt word cloud (left) and category breakdown across five core audio-visual domains (right).</sub>
<br><br>
<img src="assets/leaderboard.png" width="95%" alt="SyncBench leaderboard">
<br><sub>Our continuous metric (main chart) cleanly ranks the six models; legacy evaluators (sub-charts) produce volatile, contradictory rankings.</sub>
</div>

### 🔍 Qualitative examples

<div align="center">
<img src="assets/qualitative_cases.png" width="95%" alt="Qualitative SyncBench examples">
<br><sub>Four SyncBench cases (motorcycle engine, wood chop, soft placement, chimes). Our metric penalizes missing acoustic events, premature/delayed responses, timbre mismatch, and event-count inconsistency — reflecting causal-semantic structure rather than low-level signal similarity.</sub>
</div>

## 📊 Evaluation

```bash
python evaluate.py \
    --weights   ./avsync_eval_weights.pt \
    --data_root /path/to/Crop_5s_resize \
    --batch_size 3
```

Reports overall pairwise ranking accuracy, the easy/medium/hard breakdown (by
GT-score gap), Pearson correlation, and MAE, and writes per-sample predictions to
`eval_results.json`. Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--precision_mode` | `bf16` | `bf16-mixed` replicates the training autocast path |
| `--attn_implementation` | `sdpa` | `flash_attention_2` matches the training kernel |
| `--batch_size` | `3` | lower to `1` if memory-constrained |

### Single-video & batch scoring

```bash
# One clip
python demo_single_video.py --weights ./avsync_eval_weights.pt --video clip.mp4

# A directory of generative-model outputs (no GT labels): per-clip scores +
# per-model mean/median, for ranking models against each other.
python score_videogen.py \
    --weights ./avsync_eval_weights.pt \
    --root    /path/to/VideoGen/outputs \
    --out     ./videogen_scores.json

# LTX nested layout ( <sample_id>/video_*.mp4 )
python score_ltx.py \
    --weights ./avsync_eval_weights.pt \
    --root    /path/to/LTX \
    --out     ./ltx_scores.json
```

### Programmatic use

```python
import torch
from avsync_eval.models.evaluator import AVSyncEvaluator

model = AVSyncEvaluator(model_name="Qwen/Qwen2.5-Omni-3B", v_fps=12, v_size=140)
ckpt = torch.load("avsync_eval_weights.pt", map_location="cpu")
model.load_eval_checkpoint(ckpt["state_dict"])
model.to_eval_device()                          # -> cuda, bf16, eval()

scores = model.score_batch([video_tensor], [audio_array])   # list[float]
```

`video_tensor`: `(frames, 3, H, W)`; `audio_array`: 1-D 16 kHz waveform. Use
`qwen_omni_utils.process_mm_info(..., use_audio_in_video=True)` to decode an mp4
into these (see `demo_single_video.py:load_clip`).

## 🔧 Training

The released checkpoint comes from a two-stage recipe: **SFT cold start → listwise
RL**. Both stages use the same entry point, `train.py`, selected by
`--train_mode`.

**Model.** `AVSyncTrainModule` (`avsync_eval/training/module.py`) wraps a *policy*
Qwen2.5-Omni Thinker + score head plus a frozen *reference* copy used by the RL
objectives. Only the LLM transformer layers and the score head are trained; the
audio/vision encoders and token embeddings stay frozen. The head layout matches
`AVSyncEvaluator` / `convert_checkpoint.py`, so any checkpoint trained here can be
converted and evaluated with the inference path above.

**Training modes.**

| Mode      | Objective                                                          | Data / item |
|-----------|-------------------------------------------------------------------|:-----------:|
| `SFT`     | Cross-entropy on the `SCORE` token + Bradley-Terry pairwise loss   | 1 pair |
| `RL`      | Pairwise GRPO (Gaussian rollouts, ranking reward on the pair)      | 1 pair |
| `RL_rank` | Listwise GRPO; reward = global ranking quality over *K* methods    | *K* methods |

`RL_rank` is auto-detected per batch (its dataset emits `num_methods`); `SFT` vs
`RL` is chosen by `--train_mode`. The listwise reward composes NDCG, Kendall τ,
Spearman ρ, Top-1, MRR, and pairwise concordance
(`avsync_eval/training/ranking_reward.py`).

### Stage 1 — SFT cold start

```bash
python train.py --train_mode SFT \
    --data_root /path/to/Crop_5s_resize \
    --exp_dir   ./runs/sft \
    --devices   6
```

**Dynamic curriculum (SFT / pairwise RL).** Training starts at lesson
`--num_lession` (default 0, an easy-heavy difficulty mix) and automatically
advances to the next, harder lesson whenever the running training accuracy
exceeds `--curriculum_threshold` (default `0.88`), progressively shifting sampling
mass from easy toward hard difficulty levels. The current lesson is logged as
`lession`. `RL_rank` has no curriculum and ignores these flags.

### Stage 2 — listwise RL from the SFT checkpoint

Convert the SFT DeepSpeed checkpoint to a flat `.pt` first (see
[Convert the checkpoint](#convert-the-checkpoint)), then:

```bash
python train.py --train_mode RL_rank \
    --data_root        /path/to/Crop_5s_resize \
    --pretrained       ./sft_weights.pt \
    --sample_list_path /path/to/Crop_5s_resize/train.txt \
    --num_methods      6 \
    --exp_dir          ./runs/rl_rank \
    --devices          6
```

**Key flags** (`python train.py --help` for the full list):

| Flag | Default | Notes |
|------|---------|-------|
| `--num_methods` | `6` | methods per sample (RL_rank); each is one forward pass — bounds GPU memory |
| `--manual_std` / `--num_rollout` / `--epison` / `--kl_weight` | | GRPO hyper-parameters |
| `--learning_rate` | `1e-6` | AdamW |
| `--batch_size` | auto | `1` for RL_rank, `8` for SFT/RL |
| `--strategy` / `--precision` / `--devices` | ZeRO-2 / bf16-mixed / 6 | Lightning Trainer |

Validation runs every epoch and logs `val/pair_accuracy` (+ easy/medium/hard);
the best checkpoint by that metric is saved to `--exp_dir`.

> **VRAM guide (RL_rank, batch_size=1):** ~4–5 methods @ 24 GB · ~6–8 @ 40 GB ·
> 11 @ 80 GB.

### 📁 Use your own dataset

> **The SynthSync dataset is not yet released** (see the [Roadmap](#-roadmap)) —
> but all training and evaluation code runs on **any custom dataset**.

Point `--data_root` at a directory shaped like this:

```
my_dataset/
├── overall_scores.json                 # {sample: {method: gt_score}}      [required]
├── valing_pairs.json                   # eval / validation pairs           [eval + val]
├── train.txt                           # sample names, one per line        [RL_rank]
├── curriculumn_SFT/level_{0..9}.json   # curriculum pairs — SFT            [SFT only]
├── curriculumn_RL/level_{0..9}.json    # curriculum pairs — pairwise RL    [RL only]
└── <method>/<sample_name>.mp4          # one clip per (method, sample)     [required]
```

Each *sample* is one video; each *method* is one audio source (a V2A model, or
`GT_A` for the real audio). Audio is read from the video track — no separate
`.wav` needed. The smallest setup (evaluation only) is just `overall_scores.json`,
`valing_pairs.json`, and the mp4s.

📖 **Full guide with exact JSON schemas, per-mode file requirements, the
custom-method-name caveat, and a minimal walk-through:
[`docs/CUSTOM_DATASET.md`](docs/CUSTOM_DATASET.md).**

## 📂 Repository Layout

```text
opensource_eval/
├── convert_checkpoint.py       # DeepSpeed ZeRO ckpt  ->  single clean .pt
├── evaluate.py                 # batch eval on a dataset -> pairwise accuracy
├── demo_single_video.py        # score one mp4
├── score_videogen.py           # batch-score a flat dir of clips (no GT)
├── score_ltx.py                # batch-score the LTX nested layout (no GT)
├── train.py                    # training entry point (SFT / RL / RL_rank)
├── requirements.txt            # inference deps
├── requirements_train.txt      # + training deps
├── assets/                     # README figures
└── avsync_eval/
    ├── models/
    │   ├── evaluator.py         # AVSyncEvaluator = Thinker + linear score head
    │   └── hacked_qwen.py       # Qwen2.5-Omni Thinker (batch-preserving forward)
    ├── data/
    │   └── dataset.py           # AV_ValDataset (single (sample, method) items)
    ├── training/
    │   ├── module.py            # AVSyncTrainModule (SFT + RL + RL_rank + val)
    │   ├── train_dataset.py     # AV_Trainset (curriculum) + AV_RLRankDataset
    │   └── ranking_reward.py    # NDCG / Kendall / Spearman / Top-1 / MRR rewards
    ├── metrics.py               # pairwise ranking accuracy (easy/medium/hard)
    ├── qwen2_5_omni/            # bundled Qwen2.5-Omni implementation
    └── qwen2_vl/                # bundled Qwen2-VL implementation
```

## 🔁 Reproducibility

- The checkpoint was trained with DeepSpeed **bf16**; `to_eval_device()` casts to
  bf16 to reproduce the online training scores. For the closest match to the
  training-time autocast numerics use `--precision_mode bf16-mixed`.
- Video sampling (`fps=12`, `140×140`, 5 s → 60 frames) and audio length
  (80 000 samples) must match training — these are the defaults everywhere.
- Training and inference share the same model, head, and forward, so a checkpoint
  trained with `train.py` evaluates identically through `evaluate.py`.

## 🧭 Roadmap

- [x] Inference / evaluation package
- [x] Full training pipeline — preference SFT (dynamic curriculum) + R-GRPO
- [x] Batch scorers for unlabeled generative-model outputs
- [ ] **SynthSync dataset** — videos + pairwise human annotations *(in preparation)*
- [ ] **Trained checkpoint** on the Hugging Face Hub *(in preparation)*
- [ ] **SyncBench** prompts & generated samples

Until the dataset is released, the code is fully usable on your
[own dataset](#-use-your-own-dataset).

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{qian2026beyond,
  title     = {Beyond Time Shifts: Adapting Omni-LLM as a Reference-Free
               Evaluator for Generative Audio-Visual Models},
  author    = {Qian, Yijie and Wang, Juncheng and Xu, Chao and Wang, Huihan and
               Feng, Yuxiang and Liu, Yang and Sun, Baigui and Liu, Yong and
               Wang, Shujun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 🙏 Acknowledgments

- Built on [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) as the multimodal backbone.
- Training uses [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed).
- R-GRPO adapts the [GRPO](https://github.com/deepseek-ai/DeepSeek-R1) framework to continuous latent policies.

## 📄 License

Released under the [MIT License](LICENSE). The Qwen2.5-Omni base weights are
subject to their own license from the model provider.
