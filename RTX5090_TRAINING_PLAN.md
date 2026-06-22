# UzSL on RTX 5090 — Data Splitting + Experiment Plan

Target metric throughout: **new-signer top-1 accuracy** (signer-independent), reported as
**mean ± std over leave-one-signer-out folds**, on the evaluable vocabulary. Random-split
accuracy is already ~91% and saturated — ignore it as a goal, it does not measure what we care about.

This doc is meant to be handed to Copilot on the 5090 machine. Sections marked **[COPILOT PROMPT]**
are ready to paste. Sections marked **[RUN]** are shell commands.

---

## 0. Your data at a glance

| Signer | Unique signs | Pose files | Coverage |
|--------|-------------:|-----------:|----------|
| s01 | 1,359 | 5,349 | full |
| s02 | 1,320 | 5,104 | full |
| s03 | 1,359 | 5,378 | full |
| s05 | 1,359 | 5,377 | full |
| s04 | 967 | 2,713 | partial |
| s08 | 835 | 3,627 | partial |
| s06 | 666 | 2,665 | partial |
| s07 | 252 | 906 | low |
| **Total** | **~1,359** | **31,119** | 8 signers |

Two facts drive every decision below:
1. **4 signers (s01/s02/s03/s05) cover the full vocabulary.** They are your reliable train backbone
   and your val/test anchors. As long as ≥2 of them are in train, every evaluable class has examples.
2. **The bottleneck is signer generalization, not capacity or data volume.** This is normal: the field
   sees a **15–34 point drop** from random split to held-out signer (AUTSL 95.95%→62.02%, LSA64
   97%→74%, KArSL 99.7%→68%). Your job is to close that gap, and the levers are known (Section 3+).

---

## 1. The splitting protocol (do this first)

### Why LOSO and not a single holdout
With only 8 signers you are in the "small dataset" regime. A single held-out test signer is one noisy
point estimate. The honest protocol used on small SLR datasets (e.g. LSA64) is **Leave-One-Signer-Out
cross-validation**: rotate each signer as test, report **mean ± std**. The std is itself a headline —
it tells you how much accuracy swings depending on *which* signer is new.

### Three rules (from AUTSL / MS-ASL / FluentSigners-50)
1. **Signer-disjoint:** every signer belongs to exactly one of {train, val, test} within a fold.
2. **Validation is a THIRD signer**, disjoint from train and test. If val is a random subset of training
   signers, early-stopping/checkpoint selection silently leaks signer identity and inflates your test
   number. Pick val from the full-coverage group so it covers most of the vocabulary.
3. **Evaluable vocabulary only.** A class present in just one signer is unlearnable when that signer is
   held out. Train on the *full* vocabulary, but **score only on classes with ≥2 signers**, and per fold
   exclude any class with 0 training rows. Report the evaluable vocab size explicitly.

### Generate the split columns — **[RUN]**
A generator script is already written for you at `scripts/uzsl_v1/make_loso_splits.py`:

```bash
python -m scripts.uzsl_v1.make_loso_splits \
  --in  experiments/new_dataset/manifests/train_manifest_available.csv \
  --out experiments/new_dataset/manifests/train_manifest_loso.csv
```

It adds: `eval_vocab` (1/0), `dev_split` (one fast fold), and `loso_s01 … loso_s08` (8 folds, each with a
high-coverage val signer auto-picked). It prints per-fold train/val/test sizes so you can sanity-check
coverage before spending GPU time. If your manifest doesn't exist yet on the 5090 box, build it first
with the prompt in Section 6A.

### Two-tier workflow
- **Tier A — dev split (`dev_split` column):** test=s05, val=s01, train=the other 6. Use this for ALL
  fast iteration / hyperparameter tuning. Touch `s05` results sparingly so you don't overfit to it.
- **Tier B — LOSO (`loso_*` columns):** run all 8 folds only for your final, headline number on a
  configuration you've already settled. On a 5090 each fold is cheap; 8 folds is an overnight job.

---

## 2. What your codebase ALREADY has (do not let Copilot rebuild these)

`scripts/uzsl_v1/train.py` already supports, via flags:

| Capability | Flag |
|---|---|
| Wrist-local hand normalization | `--wrist-norm` |
| Body-proportion (arm-length) augmentation | `--body-proportion-aug` |
| Class-balanced sampling | on by default (`--no-weighted-sampling` to disable) |
| Signer-balanced sampling | `--signer-balanced-sampling` |
| Gradient-reversal signer-adversarial head | `--signer-adversarial --signer-adversarial-weight W` |
| Slovo/RSL encoder transfer (sequential) | `--pretrained-encoder PATH --freeze-encoder-epochs N --finetune-lr-scale S` |
| Mixup + label smoothing | `--mixup-alpha`, `--mixup-p`, `--label-smoothing` |
| Cosine-prototype classifier eval | `--eval-prototypes` |
| Test-time augmentation (mirror) | `--tta-views 2` |
| AMP / torch.compile | `--amp`, `--compile` |
| Model size | `--hidden-dim`, `--n-layers`, `--n-heads`, `--dropout` |

The augmentation stack (`augment.py`) already does time-stretch, temporal crop, horizontal flip with L/R
swap, affine rotate±15°/scale, jitter, temporal mask, landmark mask. Features (`features.py`) already do
shoulder normalization + kinematics (velocity/accel) on a `hands_pose` (75-landmark) subset.

**One small code change needed** for LOSO: `--split-column` is hard-restricted to
`["split", "signer_holdout_split"]`. See Section 6B to open it up to the new `dev_split`/`loso_*` columns.

---

## 3. The experiment ladder (ranked by evidence × effort)

Each rung is measured on the **dev split** against the rung below it. Keep what helps, drop what doesn't.
On a 32 GB 5090 you can run `--hidden-dim 384 --batch-size 256 --amp --compile` comfortably — large
batch stabilizes the schedule and lets OneCycle peak higher.

### E0 — Strong baseline on the new data **[RUN]**
Establish the new-signer number on the larger dataset with everything we already trust.

```bash
python -m scripts.uzsl_v1.train \
  --manifest experiments/new_dataset/manifests/train_manifest_loso.csv \
  --data-dir <POSE_DIR> --artifact-dir artifacts/e0_baseline \
  --split-column dev_split \
  --wrist-norm --signer-balanced-sampling \
  --hidden-dim 384 --n-layers 4 --n-heads 4 \
  --epochs 120 --batch-size 256 --lr 1.5e-3 --amp --compile --device cuda --no-progress
```

### E1 — GISLR-winner feature recipe (highest evidence)
The Kaggle GISLR 1st place (same input as you: MediaPipe Holistic, signer-independent hidden test) used:
**lips (40) + hands (21+21) + nose/eyes landmarks; 6 channels/node = (x,y)+velocity+acceleration;
per-sequence z-score normalization; handedness canonicalization (mirror to a canonical hand).**
You already have velocity/accel. Add **lips landmarks** and a **per-sequence z-score option**.
→ Copilot prompt 6C. Expected: the single biggest feature-side gain.

### E2 — Scale the model (5090 makes this free)
Sweep `--hidden-dim {256,384,512}`, `--n-layers {4,6}`, `--batch-size {128,256,512}`. On 4 GB we were
capped at hidden 192; that cap is gone. Watch for overfit on the dev val signer — pick by val, not train.

### E3 — Signer-adversarial GRL (most direct cross-signer lever)
Already implemented. Sweep the weight; too high collapses training.

```bash
python -m scripts.uzsl_v1.train ... --split-column dev_split \
  --wrist-norm --signer-balanced-sampling \
  --signer-adversarial --signer-adversarial-weight 0.05 \
  --hidden-dim 384 --epochs 120 --batch-size 256 --amp --compile --device cuda --no-progress
```
Try weights {0.02, 0.05, 0.1}. Literature: DANN-style signer removal is the dominant academic
gap-closer; motion channels (E1) are the theoretical complement.

### E4 — Harden augmentation
Research finding: **combined rotate+scale is super-additive (~+5% vs ~+1% each alone)**; **joint/channel
masking and Gaussian noise are the highest-value skeleton augs**; **raw Mixup can hurt skeletons — use
manifold (feature-space) mixup instead.** You already have rotate+scale+masking. Add **manifold mixup**
(mix hidden activations, not raw inputs) → Copilot prompt 6D, and widen affine ranges.

### E5 — Transfer from a large SL dataset
You have `artifacts/slovo_pretrain/encoder.pt` (RSL). Sequential fine-tune already works:
```bash
python -m scripts.uzsl_v1.train ... --split-column dev_split \
  --pretrained-encoder artifacts/slovo_pretrain/encoder.pt \
  --freeze-encoder-epochs 5 --finetune-lr-scale 0.1 \
  --wrist-norm --hidden-dim 384 --epochs 120 --batch-size 256 --amp --device cuda --no-progress
```
**Stronger option (research-backed): CO-TRAIN, don't sequentially fine-tune.** A shared encoder with
per-language classification heads (Uzbek + Slovo/WLASL) beat sequential pretrain→finetune in the Logos
study (WLASL 66.8% vs 65.6%). This is a build → Copilot prompt 6E. Pretraining gains are largest exactly
in your low-data regime (SignBERT +13 pts, MASA +23 pts on small splits).

### E6 — Second stream + ensemble (when single-model plateaus)
Skeleton-only SOTA is an **SL-GCN / DSTA-GCN** with a **4-stream (joint / bone / joint-motion /
bone-motion)** ensemble (AUTSL 96.47% skeleton-only). Add a GCN branch on the hand+pose graph as a
second model, then **average logits (not softmax) — a consistent +0.01 trick from GISLR 2nd place.**
→ Copilot prompt 6F. Highest ceiling, highest effort.

### E7 — Self-supervised masked pretraining (biggest low-data lever, biggest build)
SignBERT/MASA: mask 40–90% of joints (or reconstruct *motion residuals*) on ALL your pose data
unlabeled, then fine-tune. +13 to +23 pts in papers, gains largest in low-data. Only attempt once
E0–E6 plateau. → Copilot prompt 6G.

**Suggested order:** E0 → E1 → E2 → E3 → E4, re-baseline, then E5; reach for E6/E7 only if you plateau.

---

## 4. Reporting (so the number is defensible)

After you lock a config on the dev split, run all 8 LOSO folds and report:
- **Headline:** mean ± std of top-1 and top-5 across the 8 `loso_*` folds, on `eval_vocab==1` classes.
- **Per-signer table** (8 rows): exposes whether one signer (likely s07 @ 252 signs, or the most
  stylistically different) drags the mean — a single split would hide this.
- **State explicitly:** vocab trained (~1,359) vs vocab evaluated (eval_vocab size), classes dropped per
  fold. Never quote a single-split point estimate as your generalization result.

LOSO driver to run all folds → Copilot prompt 6H.

---

## 5. Quick reference — the runs that matter most

```bash
# 1. Build splits
python -m scripts.uzsl_v1.make_loso_splits \
  --in experiments/new_dataset/manifests/train_manifest_available.csv \
  --out experiments/new_dataset/manifests/train_manifest_loso.csv

# 2. E0 baseline (dev split)         -> artifacts/e0_baseline
# 3. E3 + GRL                        -> artifacts/e3_grl
# 4. Winner config, all 8 LOSO folds -> artifacts/loso_<signer>/  (prompt 6H)
```

---

## 6. Copilot prompts (paste these)

### 6A — Build the manifest from the pose directory (only if you don't have one)
> I have a folder of MediaPipe `.pose` files named `<signer>_<sign>_<rep>.pose`, e.g.
> `s01_sgn_0459_r03.pose`. Write a Python script `scripts/uzsl_v1/build_manifest.py` that scans a pose
> directory, parses signer_id / sign_id / rep_id from each filename, and writes a CSV with columns:
> `sample_id, sign_id, signer_id, rep_id, pose_path` (pose_path relative to the data dir, using forward
> slashes). Skip zero-byte files and print a count of skipped files. Sort rows by signer_id then sign_id
> then rep_id. Use only the standard library (csv, pathlib, argparse).

### 6B — Open up `--split-column` for LOSO **(required for the plan)**
> In `scripts/uzsl_v1/train.py`, the argparse for `--split-column` is restricted with
> `choices=["split", "signer_holdout_split"]`. Remove the `choices=` restriction so any column name is
> accepted (keep default `"split"`). Then find where the code splits rows by this column into
> train/val/test — confirm it reads `row[args.split_column]` and treats values `"train"/"val"/"test"`.
> Also add: after building the label set from TRAIN rows only, filter val/test rows to classes present in
> train (drop unlearnable classes for that fold) and, if a column named `eval_vocab` exists, additionally
> restrict the val/test metric computation to rows where `eval_vocab == "1"`. Print how many val/test rows
> and classes were dropped. Do not change any other behavior.

### 6C — Add lips landmarks + per-sequence z-score (E1)
> In `scripts/uzsl_v1/features.py`, the current `hands_pose` component uses a 75-landmark subset (hands +
> upper-body pose) with shoulder normalization and velocity/acceleration kinematics. Add a new component
> option `hands_pose_lips` that additionally includes the ~40 MediaPipe Holistic lip landmarks. Also add a
> normalization mode `--feature-norm {shoulder,zscore}` to train.py: `zscore` applies per-sequence z-score
> normalization (subtract per-clip mean, divide by per-clip std over non-NaN values, then NaN→0) instead
> of shoulder normalization. This mirrors the GISLR Kaggle 1st-place recipe. Keep `shoulder` as default.
> Update `feature_dim()` accordingly and make sure the model's `frame_dim` is derived from it.

### 6D — Manifold mixup (E4)
> In `scripts/uzsl_v1/train.py` we currently do input-space mixup. Add an alternative `--manifold-mixup`
> flag that instead mixes the hidden activations at a randomly chosen block of the ConvTransformer
> (feature-space mixup) with the same Beta(alpha, alpha) lambda and mixes the labels identically. Research
> shows raw mixup can hurt skeleton data while manifold mixup helps. When `--manifold-mixup` is set, disable
> input-space mixup. This requires the model's forward to optionally take a mixup (layer_index, lambda,
> permutation) and apply it after that block. Keep the default path unchanged.

### 6E — Multi-dataset co-training with per-language heads (E5, stronger transfer)
> Build co-training in `scripts/uzsl_v1/`. Goal: one shared ConvTransformer encoder + TWO classification
> heads — one for UzSL (~1,359 classes), one for the source dataset (Slovo RSL, ~1,000 classes, landmarks
> already cached for `slovo_data.py`). Each minibatch draws from both datasets (alternate or concatenate);
> each sample is routed to its own head; losses are summed (optionally weighted). This follows the Logos
> finding that co-training a shared encoder with per-language heads beats sequential pretrain→finetune for
> a small target language. Reuse the existing encoder from `model.py` and the existing augmentation. Add a
> script `scripts/uzsl_v1/cotrain.py` with flags for both manifests, head sizes, and a `--source-loss-weight`.
> Evaluate ONLY the UzSL head on the UzSL signer-independent val/test split.

### 6F — SL-GCN second stream + logit ensemble (E6)
> Add a skeleton GCN model (`scripts/uzsl_v1/model_gcn.py`) operating on the hand+pose graph: build the
> adjacency for the 21+21 hand joints + upper-body pose joints, implement a decoupled spatial-temporal GCN
> (SL-GCN / ST-GCN style) with the same input frames as the ConvTransformer. Train it with the same
> train.py data pipeline and splits (add `--architecture sl_gcn`). Then add an ensemble eval script that
> loads a trained conv_transformer checkpoint AND an sl_gcn checkpoint and reports signer-independent
> accuracy from AVERAGED LOGITS (not averaged softmax — averaging logits gave a consistent boost in GISLR).
> Optionally support 4 streams (joint / bone / joint-motion / bone-motion) by deriving bone = child−parent
> and motion = temporal diff, training one model per stream, and averaging all logits.

### 6G — Self-supervised masked pretraining (E7)
> Implement SignBERT/MASA-style self-supervised pretraining in `scripts/uzsl_v1/ssl_pretrain.py`: on ALL
> pose clips (labels ignored), randomly mask 40–90% of joints per frame and train the ConvTransformer
> encoder + a lightweight decoder to reconstruct either the masked joint coordinates OR the motion
> residuals (temporal differences) at masked positions, using MSE over masked entries only. Save the
> encoder weights in the same format `--pretrained-encoder` expects. Then fine-tune with the existing
> train.py path. Reconstructing motion residuals is the variant with the strongest reported cross-signer
> gains.

### 6H — LOSO driver: run all 8 folds and aggregate (reporting)
> Write `scripts/uzsl_v1/run_loso.py` that, given a base train.py command and the loso manifest, runs
> training once per `loso_<signer>` column (s01…s08), each into `artifacts/loso_<signer>/`, then reads
> each fold's `metrics.json`, and prints a table of per-signer test top-1/top-5 plus the mean ± std across
> folds (computed only over `eval_vocab==1` classes). Also write the aggregate to
> `artifacts/loso_summary.json`. Folds can run sequentially (one 5090). Pass through `--hidden-dim`,
> `--wrist-norm`, etc. unchanged to each fold.
