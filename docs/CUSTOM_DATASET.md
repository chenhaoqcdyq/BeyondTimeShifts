# Using Your Own Dataset

Our **SynthSync** dataset (videos + human annotations) is available at
[🤗 qianyijie/leaderboard](https://huggingface.co/qianyijie/leaderboard). You can
also run all training and evaluation code on **any custom dataset** that follows
the layout below.

**Terminology.** Each *sample* is one underlying video. Each *method* is one audio
source — a video-to-audio (V2A) model, or `GT_A` for the real recorded audio —
that produced a clip for that video. So `mmaudio/clip0001.mp4` is method `mmaudio`
applied to sample `clip0001`.

---

## Directory layout

Point `--data_root` at a directory shaped like this:

```
my_dataset/
├── overall_scores.json                 # {sample: {method: gt_score}}      [required]
├── valing_pairs.json                   # validation / eval comparison pairs [eval + val]
├── train.txt                           # sample names, one per line         [RL_rank]
├── curriculumn_SFT/level_{0..9}.json   # curriculum pairs — SFT             [SFT only]
├── curriculumn_RL/level_{0..9}.json    # curriculum pairs — pairwise RL     [RL only]
└── <method>/<sample_name>.mp4          # one clip per (method, sample)      [required]
```

Every `<method>/<sample_name>.mp4` is a clip **with an audio track** — the audio is
read directly from the video, so no separate `.wav` file is needed. Clips are
automatically padded/trimmed at load time to 5 s (60 frames @ 12 FPS, `140×140`)
and 80 000 audio samples, so your source clips do not need to be pre-trimmed.

### Which files each mode needs

| Task | Required files |
|------|----------------|
| **Evaluate** a checkpoint | `overall_scores.json`, `valing_pairs.json`, `<method>/*.mp4` |
| **RL_rank** (listwise) | `overall_scores.json`, `train.txt`, `<method>/*.mp4` (+ `valing_pairs.json` for validation) |
| **SFT** (cold start) | above **+** `curriculumn_SFT/level_{0..9}.json` |
| **RL** (pairwise) | above **+** `curriculumn_RL/level_{0..9}.json` |

---

## File formats

### `overall_scores.json` — ground-truth scores *(required)*

The single source of truth for scores. Maps each sample to a `{method: score}`
dict, where a higher score means better synchronization. Scores can be any real
numbers (pairwise win-rates, MOS, human ratings, …); **only their relative
ordering within a sample is used**, never the absolute values.

```json
{
  "clip0001": {"mmaudio": 0.42, "foley": 0.19, "GT_A": 1.0},
  "clip0002": {"mmaudio": 0.16, "foley": 0.55, "GT_A": 1.0}
}
```

### `valing_pairs.json` — comparison pairs *(evaluation + training validation)*

Maps each sample to a list of method pairs to compare. Each pair is
`[method1, method2]`; a 3rd element (e.g. a score gap) is **optional and ignored
by the code** — scores are always looked up from `overall_scores.json`. Pairs
whose two methods have equal GT scores are skipped.

```json
{
  "clip0001": [["mmaudio", "foley"], ["mmaudio", "GT_A"]],
  "clip0002": [["foley", "GT_A"]]
}
```

Evaluation reports overall pairwise accuracy plus an **easy / medium / hard**
breakdown, bucketed by the GT-score gap of each pair:

| Bucket | GT-score gap |
|--------|--------------|
| easy   | > 0.5 |
| medium | 0.1 – 0.5 |
| hard   | ≤ 0.1 |

### `train.txt` — training sample list *(RL_rank)*

Plain text, one sample name per line. Used only by `RL_rank` to pick which samples
to draw from.

```
clip0001
clip0002
```

### `curriculumn_SFT/level_{i}.json` & `curriculumn_RL/level_{i}.json` — curriculum *(SFT / pairwise RL)*

Needed only for `SFT` and pairwise `RL`. Provide ten files per mode
(`level_0.json` … `level_9.json`), one per difficulty bucket, where **`level_0` is
the hardest** (smallest score gap between the paired methods) and **`level_9` the
easiest**. The training curriculum starts easy and advances toward harder buckets
as accuracy improves.

Each file maps a sample to a list of method pairs. **Only the first two elements
(the two method names) are read** — any trailing values (scores, gaps) are ignored
and may be omitted.

```json
{
  "clip0001": [["mmaudio", "foley"], ["foley", "GT_A"]]
}
```

> **Tip — building the curriculum.** For each sample, rank its methods by
> `overall_scores.json`, then place a pair `(a, b)` into `level_i` according to how
> far apart `a` and `b` are in that ranking: large rank gaps → easy levels (high
> `i`), small gaps → hard levels (low `i`). This mirrors how SynthSync was built.

---

## ⚠️ Custom method names

The listwise `AV_RLRankDataset` and the curriculum `AV_Trainset` reference a fixed
method list, `ALL_METHODS`, in
[`avsync_eval/training/train_dataset.py`](../avsync_eval/training/train_dataset.py):

```python
ALL_METHODS = [
    "foleycontrol", "audiox", "cafa", "foley", "hunyuan",
    "lova", "melqcd", "mmaudio", "selva", "vta_ldm",
]   # "GT_A" (real audio) is appended when include_gt=True
```

If your dataset uses **different method names**, edit this list to match. Every
name in `ALL_METHODS` must appear both:

1. as a `<method>/` sub-directory under `--data_root`, and
2. as a key in each sample's `overall_scores.json` entry.

Evaluation (`evaluate.py`) is unaffected — it reads method names directly from
`valing_pairs.json`, so it works with any names without code changes.

---

## Minimal walk-through (evaluation only)

The smallest possible setup to score an existing checkpoint on your own clips:

```
my_dataset/
├── overall_scores.json     # {"clip0001": {"modelA": 0.4, "modelB": 0.2}}
├── valing_pairs.json       # {"clip0001": [["modelA", "modelB"]]}
├── modelA/clip0001.mp4
└── modelB/clip0001.mp4
```

```bash
python evaluate.py --weights ./avsync_eval_weights.pt --data_root ./my_dataset
```

For unlabeled clips (no `overall_scores.json`), use the batch scorers instead —
`tools/score_videogen.py` (flat directory) or `tools/score_ltx.py` (nested layout)
— which emit per-clip scores and per-model aggregates. See the
[main README](../README.md#single-video--batch-scoring).
