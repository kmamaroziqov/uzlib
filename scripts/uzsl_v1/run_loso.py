"""Run all 8 leave-one-signer-out folds and aggregate results.

Usage (from repo root):
  python -m scripts.uzsl_v1.run_loso \
      --manifest experiments/new_dataset/manifests/train_manifest_loso.csv \
      --data-dir D:/uzsl_data_local \
      --out-dir  artifacts/loso_run1 \
      --hidden-dim 384 --epochs 120 --batch-size 256 --amp --compile --device cuda \
      --wrist-norm --signer-balanced-sampling

Every flag not listed above is passed through unchanged to train.py for each fold.
--split-column is set automatically to loso_<signer> for each fold; do not pass it.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


SIGNERS = ["s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08"]


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run LOSO cross-validation across all signer folds."
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--signers", nargs="+", default=SIGNERS)
    args, passthrough = ap.parse_known_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for signer in args.signers:
        fold_col = f"loso_{signer}"
        fold_dir = args.out_dir / f"fold_{signer}"
        print(f"\n{'='*60}", flush=True)
        print(f"  FOLD: {signer} as TEST  ({fold_col})", flush=True)
        print(f"{'='*60}", flush=True)

        cmd = [
            sys.executable, "-u", "-m", "scripts.uzsl_v1.train",
            "--manifest", str(args.manifest),
            "--data-dir", str(args.data_dir),
            "--artifact-dir", str(fold_dir),
            "--split-column", fold_col,
            "--no-progress",
            *passthrough,
        ]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"WARNING: fold {signer} exited with code {ret.returncode}", flush=True)

        metrics_file = fold_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as fh:
                m = json.load(fh)
            val_top1  = m.get("val",  {}).get("top1",  None)
            test_top1 = m.get("test", {}).get("top1",  None)
            val_top5  = m.get("val",  {}).get("top5",  None)
            test_top5 = m.get("test", {}).get("top5",  None)
            results.append({
                "signer": signer,
                "val_top1": val_top1,
                "test_top1": test_top1,
                "val_top5": val_top5,
                "test_top5": test_top5,
            })
        else:
            print(f"WARNING: no metrics.json found for fold {signer}", flush=True)
            results.append({"signer": signer, "val_top1": None, "test_top1": None})

    # Aggregate
    test_top1_vals = [r["test_top1"] for r in results if r["test_top1"] is not None]
    test_top5_vals = [r["test_top5"] for r in results if r["test_top5"] is not None]
    mean1, std1 = mean_std(test_top1_vals)
    mean5, std5 = mean_std(test_top5_vals)

    summary = {
        "folds": results,
        "test_top1_mean": mean1,
        "test_top1_std":  std1,
        "test_top5_mean": mean5,
        "test_top5_std":  std5,
        "n_folds": len(test_top1_vals),
    }
    out_file = args.out_dir / "loso_summary.json"
    with out_file.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"  LOSO SUMMARY  ({len(test_top1_vals)}/{len(args.signers)} folds complete)")
    print(f"{'='*60}")
    print(f"  {'Signer':<8}  {'val_top1':>10}  {'test_top1':>10}  {'test_top5':>10}")
    for r in results:
        v1 = f"{r['val_top1']:.4f}" if r['val_top1'] is not None else "  N/A  "
        t1 = f"{r['test_top1']:.4f}" if r['test_top1'] is not None else "  N/A  "
        t5 = f"{r['test_top5']:.4f}" if r['test_top5'] is not None else "  N/A  "
        print(f"  {r['signer']:<8}  {v1:>10}  {t1:>10}  {t5:>10}")
    print(f"  {'mean':<8}  {'':>10}  {mean1:>10.4f}  {mean5:>10.4f}")
    print(f"  {'std':<8}  {'':>10}  {std1:>10.4f}  {std5:>10.4f}")
    print(f"\n  Saved -> {out_file}")


if __name__ == "__main__":
    main()
