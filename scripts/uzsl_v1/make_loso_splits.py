"""Generate signer-independent (signer-disjoint) split columns for the UzSL manifest.

Produces, in a copy of the manifest:
  * eval_vocab           : 1/0 flag — class appears in >= MIN_SIGNERS_PER_CLASS signers
                           (the only classes on which signer-independent accuracy is honest)
  * dev_split            : a single fast-iteration split (train / val / test), signer-disjoint
  * loso_<signer>        : one column per signer fold; that signer = test, a high-coverage
                           signer = val, everyone else = train. Nested + signer-disjoint.

Why this design (grounded in AUTSL / MS-ASL / FluentSigners-50 conventions):
  - With only 8 signers you are in the "small dataset" regime -> Leave-One-Signer-Out (LOSO)
    is the honest protocol, not a single fixed holdout. Report mean +/- std over folds.
  - Validation must be a THIRD signer, disjoint from both train and test, or model selection
    leaks signer identity and inflates the test number.
  - A class present in only ONE signer becomes unlearnable when that signer is held out
    (0 training examples). We train on the FULL vocabulary but only SCORE on the evaluable
    vocabulary (classes with >= 2 signers), and per fold we additionally exclude any class
    left with 0 training rows. The training scripts already restrict classes to those seen
    in train, so the unlearnable classes simply never enter the label set for that fold.

Usage:
  python -m scripts.uzsl_v1.make_loso_splits \
      --in  experiments/new_dataset/manifests/train_manifest_available.csv \
      --out experiments/new_dataset/manifests/train_manifest_loso.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

# Classes must appear in at least this many distinct signers to be "evaluable".
MIN_SIGNERS_PER_CLASS = 2

# Signers with (near) full vocabulary coverage. Validation folds are drawn from here so
# model selection sees most of the vocabulary and stays stable across folds.
# Adjust to match your dataset's coverage table if signer IDs differ.
HIGH_COVERAGE = ["s01", "s02", "s03", "s05"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def pick_val_signer(test_signer: str, all_signers: list[str]) -> str:
    """Choose a high-coverage validation signer that is not the test signer."""
    for cand in HIGH_COVERAGE:
        if cand != test_signer and cand in all_signers:
            return cand
    # Fallback: any other signer
    for cand in all_signers:
        if cand != test_signer:
            return cand
    raise ValueError("Need at least two signers to build a split.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="out", type=Path, required=True)
    ap.add_argument("--signer-col", default="signer_id")
    ap.add_argument("--sign-col", default="sign_id")
    ap.add_argument("--dev-test", default="s05", help="Held-out TEST signer for the fast dev split.")
    ap.add_argument("--dev-val", default="s01", help="Held-out VAL signer for the fast dev split.")
    args = ap.parse_args()

    rows = read_rows(args.inp)
    signers = sorted({r[args.signer_col] for r in rows})
    print(f"signers: {signers}")

    # --- evaluable vocabulary: classes seen in >= MIN_SIGNERS_PER_CLASS signers ---
    signers_per_class: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        signers_per_class[r[args.sign_col]].add(r[args.signer_col])
    eval_vocab = {c for c, s in signers_per_class.items() if len(s) >= MIN_SIGNERS_PER_CLASS}
    total_classes = len(signers_per_class)
    print(
        f"classes: {total_classes} total, {len(eval_vocab)} evaluable "
        f"(>= {MIN_SIGNERS_PER_CLASS} signers), "
        f"{total_classes - len(eval_vocab)} single-signer (train-only)"
    )

    # --- build the per-fold + dev split columns ---
    loso_cols = [f"loso_{s}" for s in signers]
    for r in rows:
        sgn = r[args.signer_col]
        r["eval_vocab"] = "1" if r[args.sign_col] in eval_vocab else "0"

        # Fast dev split (single fold): one test signer, one val signer, rest train.
        if sgn == args.dev_test:
            r["dev_split"] = "test"
        elif sgn == args.dev_val:
            r["dev_split"] = "val"
        else:
            r["dev_split"] = "train"

        # One LOSO column per fold.
        for test_signer in signers:
            val_signer = pick_val_signer(test_signer, signers)
            col = f"loso_{test_signer}"
            if sgn == test_signer:
                r[col] = "test"
            elif sgn == val_signer:
                r[col] = "val"
            else:
                r[col] = "train"

    # --- report per-fold sizes so you can see coverage before training ---
    print("\nfold sizes (train / val / test rows):")
    for test_signer in signers:
        col = f"loso_{test_signer}"
        c = Counter(r[col] for r in rows)
        val_signer = pick_val_signer(test_signer, signers)
        print(
            f"  {col:12s} val={val_signer}  "
            f"train={c['train']:6d}  val={c['val']:5d}  test={c['test']:5d}"
        )

    fieldnames = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    print(f"new columns: eval_vocab, dev_split, {', '.join(loso_cols)}")


if __name__ == "__main__":
    main()
