from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .paths import DEFAULT_DATA_DIR, DEFAULT_MANIFEST
from .pose_io import read_pose_file


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_pose_path(data_dir: Path, pose_path: str) -> Path:
    path = Path(pose_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == data_dir.name:
        return data_dir.parent / path
    return data_dir / path


def validate_manifest(
    manifest_path: Path,
    data_dir: Path,
    *,
    require_poses: bool = False,
    parse_poses: bool = False,
) -> dict[str, int]:
    rows = read_manifest(manifest_path)
    errors: list[str] = []
    split_counts = Counter()
    sign_ids = set()

    for row in rows:
        sample_id = row.get("sample_id", "<missing>")
        sign_id = row.get("sign_id", "")
        sign_ids.add(sign_id)
        split_counts[row.get("split", "")] += 1

        for field in ("sample_id", "sign_id", "label_uz", "video_path", "pose_path", "split"):
            if not row.get(field):
                errors.append(f"{sample_id}: missing {field}")

        video_path = data_dir / row.get("video_path", "")
        if not video_path.exists():
            errors.append(f"{sample_id}: missing video {video_path}")

        pose_path = resolve_pose_path(data_dir, row.get("pose_path", ""))
        if require_poses and not pose_path.exists():
            errors.append(f"{sample_id}: missing pose {pose_path}")
        if parse_poses and pose_path.exists():
            try:
                read_pose_file(pose_path)
            except Exception as exc:
                errors.append(f"{sample_id}: invalid pose {pose_path}: {exc}")

    if errors:
        preview = "\n".join(errors[:25])
        suffix = "" if len(errors) <= 25 else f"\n... {len(errors) - 25} more errors"
        raise SystemExit(f"Validation failed with {len(errors)} error(s):\n{preview}{suffix}")

    return {
        "rows": len(rows),
        "signs": len(sign_ids),
        "train": split_counts["train"],
        "val": split_counts["val"],
        "test": split_counts["test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UzSL v1 manifest, videos, and generated poses.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--require-poses", action="store_true")
    parser.add_argument("--parse-poses", action="store_true")
    args = parser.parse_args()

    stats = validate_manifest(
        args.manifest,
        args.data_dir,
        require_poses=args.require_poses,
        parse_poses=args.parse_poses,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("status: ok")


if __name__ == "__main__":
    main()
