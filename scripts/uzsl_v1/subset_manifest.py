from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .paths import DEFAULT_MANIFEST


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_subset(
    manifest: Path,
    output: Path,
    *,
    max_signs: int,
    exclude_category: set[str],
    min_samples_per_sign: int,
) -> dict[str, int]:
    rows = read_manifest(manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {manifest}")

    by_sign: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("category", "") in exclude_category:
            continue
        by_sign[row["sign_id"]].append(row)

    selected_signs: list[str] = []
    for sign_id in sorted(by_sign):
        if len(by_sign[sign_id]) < min_samples_per_sign:
            continue
        selected_signs.append(sign_id)
        if len(selected_signs) >= max_signs:
            break

    selected = set(selected_signs)
    subset_rows = [row for row in rows if row["sign_id"] in selected]
    write_manifest(output, subset_rows, list(rows[0].keys()))

    split_counts = Counter(row["split"] for row in subset_rows)
    signer_counts = Counter(row["signer_id"] for row in subset_rows)
    return {
        "signs": len(selected_signs),
        "samples": len(subset_rows),
        "train": split_counts["train"],
        "val": split_counts["val"],
        "test": split_counts["test"],
        "signers": len(signer_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reusable subset manifest for UzSL experiments.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("uzsl_data/generated/manifests/words_100_manifest.csv"))
    parser.add_argument("--max-signs", type=int, default=100)
    parser.add_argument("--exclude-category", action="append", default=["alifbo"])
    parser.add_argument("--min-samples-per-sign", type=int, default=1)
    args = parser.parse_args()

    stats = make_subset(
        args.manifest,
        args.output,
        max_signs=args.max_signs,
        exclude_category=set(args.exclude_category),
        min_samples_per_sign=args.min_samples_per_sign,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"manifest: {args.output}")


if __name__ == "__main__":
    main()
