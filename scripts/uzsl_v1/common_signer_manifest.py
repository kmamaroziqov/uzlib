from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_signers(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_common_signer_manifest(
    source: Path,
    output: Path,
    *,
    min_signers: int,
    train_signers: list[str],
    val_signer: str,
    test_signers: list[str],
    require_val: bool,
    require_all_test: bool,
) -> dict[str, object]:
    rows = read_rows(source)
    if not rows:
        raise ValueError(f"{source} has no rows")

    split_signers = set(train_signers) | {val_signer} | set(test_signers)
    if len(split_signers) != len(train_signers) + 1 + len(test_signers):
        raise ValueError("Train, val, and test signers must be disjoint")

    by_sign: dict[str, list[dict[str, str]]] = defaultdict(list)
    signer_coverage: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sign_id = row["sign_id"]
        signer_id = row.get("signer_id", "")
        by_sign[sign_id].append(row)
        signer_coverage[sign_id].add(signer_id)

    keep_signs: set[str] = set()
    for sign_id, coverage in signer_coverage.items():
        if len(coverage) < min_signers:
            continue
        if not any(signer in coverage for signer in train_signers):
            continue
        if require_val and val_signer not in coverage:
            continue
        if require_all_test and not all(signer in coverage for signer in test_signers):
            continue
        if not require_all_test and not any(signer in coverage for signer in test_signers):
            continue
        keep_signs.add(sign_id)

    out_rows: list[dict[str, str]] = []
    for row in rows:
        if row["sign_id"] not in keep_signs:
            continue
        signer_id = row.get("signer_id", "")
        if signer_id in train_signers:
            split = "train"
        elif signer_id == val_signer:
            split = "val"
        elif signer_id in test_signers:
            split = "test"
        else:
            continue
        out = dict(row)
        out["split"] = split
        out["signer_holdout_split"] = split
        out_rows.append(out)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(out_rows)

    split_counts = Counter(row["signer_holdout_split"] for row in out_rows)
    signer_counts = Counter(row.get("signer_id", "") for row in out_rows)
    split_classes: dict[str, set[str]] = defaultdict(set)
    for row in out_rows:
        split_classes[row["signer_holdout_split"]].add(row["sign_id"])

    return {
        "source_rows": len(rows),
        "source_signs": len(by_sign),
        "kept_rows": len(out_rows),
        "kept_signs": len(keep_signs),
        "min_signers": min_signers,
        "train_signers": train_signers,
        "val_signer": val_signer,
        "test_signers": test_signers,
        "require_val": require_val,
        "require_all_test": require_all_test,
        "split_rows": dict(sorted(split_counts.items())),
        "split_classes": {name: len(values) for name, values in sorted(split_classes.items())},
        "signer_rows": dict(sorted(signer_counts.items())),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a common-sign signer-disjoint manifest.")
    parser.add_argument("--source", type=Path, default=Path("uzsl_data_v3/generated/manifests/train_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("uzsl_data_v3/generated/manifests/common_5train_s02_val_s06_test_manifest.csv"))
    parser.add_argument("--min-signers", type=int, default=5)
    parser.add_argument("--train-signers", default="s01,s03,s04,s05,s07")
    parser.add_argument("--val-signer", default="s02")
    parser.add_argument("--test-signers", default="s06")
    parser.add_argument("--allow-missing-val", dest="require_val", action="store_false")
    parser.add_argument("--require-all-test", action="store_true")
    parser.set_defaults(require_val=True, require_all_test=False)
    args = parser.parse_args()

    stats = build_common_signer_manifest(
        args.source,
        args.output,
        min_signers=args.min_signers,
        train_signers=parse_signers(args.train_signers),
        val_signer=args.val_signer,
        test_signers=parse_signers(args.test_signers),
        require_val=args.require_val,
        require_all_test=args.require_all_test,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
