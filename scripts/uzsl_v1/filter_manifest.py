"""Build an available manifest containing only rows whose .pose file exists."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .validate_data import resolve_pose_path


def filter_manifest(source: Path, output: Path, data_dir: Path) -> dict[str, object]:
    with source.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    available = [row for row in rows if resolve_pose_path(data_dir, row["pose_path"]).exists()]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(available)

    splits = Counter(row["split"] for row in available)
    holdout = Counter(row["signer_holdout_split"] for row in available)
    signers = Counter(row["signer_id"] for row in available)
    return {
        "source_rows": len(rows),
        "available_rows": len(available),
        "unique_signs": len({row["sign_id"] for row in available}),
        "signers": dict(sorted(signers.items())),
        "random_split": {"train": splits["train"], "val": splits["val"], "test": splits["test"]},
        "holdout_split": {
            "train": holdout["train"],
            "val": holdout["val"],
            "test": holdout["test"],
            "unlabelled": holdout[""],
        },
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only manifest rows with existing pose files.")
    parser.add_argument("--source", type=Path, default=Path("uzsl_data/generated/manifests/train_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("uzsl_data/generated/manifests/available_manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("uzsl_data"))
    args = parser.parse_args()

    stats = filter_manifest(args.source, args.output, args.data_dir)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
