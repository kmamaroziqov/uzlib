from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from .paths import DEFAULT_DATA_DIR, DEFAULT_MANIFEST, DEFAULT_POSE_DIR, DEFAULT_UNSUPPORTED, video_stem


MANIFEST_FIELDS = [
    "sample_id",
    "sign_id",
    "label_uz",
    "label_ru",
    "category",
    "category_ru",
    "signer_id",
    "rep_id",
    "video_path",
    "pose_path",
    "split",
    "signer_holdout_split",
    "sign_type",
]

UNSUPPORTED_FIELDS = [
    "sign_id",
    "label_uz",
    "label_ru",
    "category",
    "category_ru",
    "sign_type",
    "reason",
]

EXCLUDED_SAMPLE_FIELDS = [
    "sample_id",
    "sign_id",
    "signer_id",
    "rep_id",
    "video_path",
    "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def stable_float(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def stratified_split(samples: list[dict[str, str]]) -> dict[str, str]:
    by_sign: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        by_sign[row["sign_id"]].append(row)

    splits: dict[str, str] = {}
    for sign_id, rows in by_sign.items():
        ordered = sorted(rows, key=lambda r: stable_float(r["sample_id"]))
        n = len(ordered)
        if n == 1:
            labels = ["train"]
        elif n == 2:
            labels = ["train", "test"]
        elif n == 3:
            labels = ["train", "val", "test"]
        else:
            n_test = max(1, round(n * 0.10))
            n_val = max(1, round(n * 0.10))
            n_train = max(1, n - n_val - n_test)
            labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
            labels = labels[:n]

        for row, split in zip(ordered, labels):
            splits[row["sample_id"]] = split

    return splits


HOLDOUT_VAL_SIGNER = "s04"
HOLDOUT_TEST_SIGNER = "s05"


def signer_holdout_split(samples: list[dict[str, str]]) -> dict[str, str]:
    # Standard signer-disjoint protocol: train on EVERYTHING from the train signers,
    # evaluate held-out signers only on classes seen in training. Holdout rows whose
    # sign never appears in training stay unlabeled (they are unpredictable by design).
    train_signs = {
        row["sign_id"]
        for row in samples
        if row["signer_id"] not in (HOLDOUT_VAL_SIGNER, HOLDOUT_TEST_SIGNER)
    }

    splits: dict[str, str] = {}
    for row in samples:
        signer = row["signer_id"]
        if signer == HOLDOUT_TEST_SIGNER:
            splits[row["sample_id"]] = "test" if row["sign_id"] in train_signs else ""
        elif signer == HOLDOUT_VAL_SIGNER:
            splits[row["sample_id"]] = "val" if row["sign_id"] in train_signs else ""
        else:
            splits[row["sample_id"]] = "train"
    return splits


def build_manifest(
    data_dir: Path,
    manifest_path: Path,
    unsupported_path: Path,
    pose_dir: Path,
    excluded_samples_path: Path | None = None,
) -> dict[str, int]:
    metadata = data_dir / "metadata"
    samples = read_csv(metadata / "samples.csv")
    signs = {row["sign_id"]: row for row in read_csv(metadata / "signs.csv")}
    categories = {row["cat_id"]: row for row in read_csv(metadata / "categories.csv")}

    valid_samples = [row for row in samples if signs.get(row["sign_id"], {}).get("label_uz")]
    sampled_sign_ids = {row["sign_id"] for row in valid_samples}
    split_by_sample = stratified_split(valid_samples)
    holdout_by_sample = signer_holdout_split(valid_samples)

    manifest_rows: list[dict[str, str]] = []
    excluded_sample_rows: list[dict[str, str]] = []
    for sample in sorted(samples, key=lambda r: r["sample_id"]):
        sign = signs.get(sample["sign_id"], {})
        cat = categories.get(sign.get("cat_id", ""), {})
        if not sign or not sign.get("label_uz"):
            excluded_sample_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "sign_id": sample["sign_id"],
                    "signer_id": sample["signer_id"],
                    "rep_id": sample["rep_id"],
                    "video_path": sample["video_path"],
                    "reason": "missing_sign_metadata",
                }
            )
            continue
        stem = video_stem(sample["video_path"])
        pose_path = pose_dir / f"{stem}.pose"
        if not pose_path.is_absolute():
            try:
                pose_path_value = pose_path.relative_to(data_dir).as_posix()
            except ValueError:
                pose_path_value = pose_path.as_posix()
        else:
            try:
                pose_path_value = pose_path.relative_to(data_dir.resolve()).as_posix()
            except ValueError:
                pose_path_value = pose_path.as_posix()
        manifest_rows.append(
            {
                "sample_id": sample["sample_id"],
                "sign_id": sample["sign_id"],
                "label_uz": sign.get("label_uz", ""),
                "label_ru": sign.get("label_ru", ""),
                "category": cat.get("name_uz", ""),
                "category_ru": cat.get("name_ru", ""),
                "signer_id": sample["signer_id"],
                "rep_id": sample["rep_id"],
                "video_path": sample["video_path"],
                "pose_path": pose_path_value,
                "split": split_by_sample.get(sample["sample_id"], "train"),
                "signer_holdout_split": holdout_by_sample.get(sample["sample_id"], ""),
                "sign_type": sign.get("sign_type", ""),
            }
        )

    unsupported_rows: list[dict[str, str]] = []
    for sign_id, sign in sorted(signs.items()):
        if sign_id in sampled_sign_ids:
            continue
        cat = categories.get(sign.get("cat_id", ""), {})
        unsupported_rows.append(
            {
                "sign_id": sign_id,
                "label_uz": sign.get("label_uz", ""),
                "label_ru": sign.get("label_ru", ""),
                "category": cat.get("name_uz", ""),
                "category_ru": cat.get("name_ru", ""),
                "sign_type": sign.get("sign_type", ""),
                "reason": "no_samples",
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    unsupported_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with unsupported_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=UNSUPPORTED_FIELDS)
        writer.writeheader()
        writer.writerows(unsupported_rows)
    if excluded_samples_path is None:
        excluded_samples_path = manifest_path.parent / "excluded_samples.csv"
    with excluded_samples_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXCLUDED_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(excluded_sample_rows)

    split_counts = Counter(row["split"] for row in manifest_rows)
    holdout_counts = Counter(row["signer_holdout_split"] for row in manifest_rows if row["signer_holdout_split"])
    return {
        "samples": len(manifest_rows),
        "sampled_signs": len(sampled_sign_ids),
        "unsupported_signs": len(unsupported_rows),
        "excluded_samples": len(excluded_sample_rows),
        "train": split_counts["train"],
        "val": split_counts["val"],
        "test": split_counts["test"],
        "holdout_train": holdout_counts["train"],
        "holdout_val": holdout_counts["val"],
        "holdout_test": holdout_counts["test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the UzSL v1 training manifest.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--unsupported", type=Path, default=DEFAULT_UNSUPPORTED)
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    parser.add_argument("--excluded-samples", type=Path, default=None)
    args = parser.parse_args()

    stats = build_manifest(args.data_dir, args.manifest, args.unsupported, args.pose_dir, args.excluded_samples)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"manifest: {args.manifest}")
    print(f"unsupported: {args.unsupported}")


if __name__ == "__main__":
    main()
