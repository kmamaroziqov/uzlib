"""One-time conversion: slovo_mediapipe.json → per-sample .npy files.

Streams the 1.2 GB JSON one video at a time (constant RAM), writing
(T, 75, 4) float32 arrays to slovo_data/poses/<attachment_id>.npy.

Run once from the project root:
    .venv/Scripts/python.exe -m scripts.uzsl_v1.convert_slovo_to_npy \
        --slovo-dir slovo_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ijson
import numpy as np

_LH_START  = 0
_RH_START  = 54
_K         = 75   # total landmarks in hands_pose layout


def _frame_to_row(frame: dict) -> np.ndarray:
    # Slovo JSON uses "hand 1", "hand 2" (no L/R label).
    # Map first detected hand → LH slot, second → RH slot.
    row = np.zeros((_K, 4), dtype=np.float32)
    hand_keys = [k for k in ("hand 1", "hand 2") if k in frame]
    starts = [_LH_START, _RH_START]
    for key, start in zip(hand_keys, starts):
        landmarks = frame[key] or []
        conf = 1.0 if landmarks else 0.0
        for k, lm in enumerate(landmarks[:21]):
            row[start + k, 0] = float(lm.get("x", 0.0))
            row[start + k, 1] = float(lm.get("y", 0.0))
            row[start + k, 2] = float(lm.get("z", 0.0))
            row[start + k, 3] = conf
    return row


def convert(slovo_dir: Path) -> None:
    json_path = slovo_dir / "slovo_mediapipe.json"
    out_dir   = slovo_dir / "poses"
    out_dir.mkdir(exist_ok=True)

    already = sum(1 for _ in out_dir.glob("*.npy"))
    if already:
        print(f"Resuming — {already} .npy files already exist, skipping them.", flush=True)

    print(f"Streaming {json_path} …", flush=True)
    done = skipped = 0

    with json_path.open("rb") as fh:
        # kvitems streams one top-level key-value pair at a time.
        # Each value is a list of frame dicts for one video.
        for vid_id, frames in ijson.kvitems(fh, ""):
            npy_path = out_dir / f"{vid_id}.npy"
            if npy_path.exists():
                skipped += 1
            else:
                rows = [_frame_to_row(f) for f in frames]
                arr = np.stack(rows) if rows else np.zeros((0, _K, 4), np.float32)
                np.save(str(npy_path), arr)
                done += 1

            total = done + skipped
            if total % 500 == 0:
                print(f"  {total} videos ({done} written, {skipped} skipped)", flush=True)

    print(f"\nDone. {done} written, {skipped} skipped. Output: {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Slovo JSON → per-sample .npy files")
    ap.add_argument("--slovo-dir", type=Path, default=Path("slovo_data"))
    args = ap.parse_args()
    convert(args.slovo_dir)


if __name__ == "__main__":
    main()
