"""Slovo RSL dataset loader for Conv1D+Transformer pretraining.

Slovo ships only LH+RH hand landmarks (no pose, no face). We map them into our
hands_pose (75-landmark) layout by zero-filling the POSE slot, and normalize
relative to the wrist-pair midpoint instead of the shoulder span.

Download:
    pip install kaggle
    kaggle datasets download kulqkul/slovo-mediapipe-json   # ~1.2 GB
    kaggle datasets download -d kapitanov/slovo -f annotations.csv

Then point --slovo-dir at the folder containing:
    slovo_mediapipe.json
    annotations.csv
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    add_kinematics,
    resample_frames,
    trim_low_activity,
    finalize,
)
TARGET_FRAMES = 64

# Module-level cache so train and val datasets share the same in-memory dict.
_LANDMARK_CACHE: dict[str, dict] = {}

# hands_pose layout: [LH(0:21), POSE(21:54), RH(54:75)]
_LH_START  = 0
_LH_END    = 21
_POSE_START = 21
_POSE_END   = 54
_RH_START  = 54
_RH_END    = 75
_K = 75    # total landmarks in hands_pose

VISIBLE_CONF = 0.05


def _wrist_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-clip normalization using wrist positions instead of shoulders.

    Center = midpoint of LH_wrist (idx 0) and RH_wrist (idx 54) over frames
    where at least one wrist is visible. Scale = mean inter-wrist distance
    when both are visible, or 1/3 of the frame if only one wrist is present.
    """
    lw = arr[:, _LH_START]    # (T, 4)  wrist of left hand
    rw = arr[:, _RH_START]    # (T, 4)  wrist of right hand
    lv = lw[:, 3] > VISIBLE_CONF
    rv = rw[:, 3] > VISIBLE_CONF
    both = lv & rv

    if both.any():
        mid = (lw[both, :3] + rw[both, :3]) / 2.0
        center = mid.mean(axis=0)
        scale = float(np.linalg.norm(lw[both, :2] - rw[both, :2], axis=1).mean())
        scale = max(scale, 1e-4)
    elif lv.any():
        center = lw[lv, :3].mean(axis=0)
        scale = 0.15  # heuristic: hand ~ 15% of frame width
    elif rv.any():
        center = rw[rv, :3].mean(axis=0)
        scale = 0.15
    else:
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 1.0

    out = arr.copy()
    out[..., :3] = (out[..., :3] - center) / scale
    # re-zero anything with zero confidence
    out[..., :3][out[..., 3] <= 0.0] = 0.0
    return out.astype(np.float32)


def _slovo_frame_to_row(frame: dict[str, Any]) -> np.ndarray:
    """Convert one Slovo frame dict → (75, 4) float32."""
    # Slovo JSON uses "hand 1", "hand 2" (no L/R label).
    # Map first detected hand → LH slot, second → RH slot.
    row = np.zeros((_K, 4), dtype=np.float32)
    hand_keys = [k for k in ("hand 1", "hand 2") if k in frame]
    starts = [_LH_START, _RH_START]
    for key, start in zip(hand_keys, starts):
        lms = frame[key] or []
        conf = 1.0 if lms else 0.0
        for k, lm in enumerate(lms[:21]):
            row[start + k, 0] = float(lm.get("x", 0.0))
            row[start + k, 1] = float(lm.get("y", 0.0))
            row[start + k, 2] = float(lm.get("z", 0.0))
            row[start + k, 3] = conf
    return row


def slovo_frames_to_array(frames_json: list[dict]) -> np.ndarray:
    """(list of per-frame dicts) → (T, 75, 4) float32."""
    return np.stack([_slovo_frame_to_row(f) for f in frames_json], axis=0)


class SlovoDataset:
    """PyTorch-compatible dataset over Slovo landmarks.

    Loads per-sample .npy files from slovo_dir/poses/ (produced by
    convert_slovo_to_npy.py). Falls back to the JSON dict if poses/ is absent
    (kept for compatibility with the smoke-test path).

    Returns (x_tensor, label_int) tuples where x is (TARGET_FRAMES, 750).
    """

    def __init__(
        self,
        slovo_dir: Path,
        *,
        split: str = "train",           # "train" or "test"
        target_frames: int = TARGET_FRAMES,
        kinematics: bool = True,
        max_classes: int | None = None,
        seed: int = 0,
        augment=None,
    ) -> None:
        slovo_dir = Path(slovo_dir)
        self.target_frames = target_frames
        self.kinematics = kinematics
        self.augment = augment
        self._poses_dir = slovo_dir / "poses"

        # ── load metadata ─────────────────────────────────────────────────
        ann_path = slovo_dir / "annotations.csv"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing {ann_path} — download from kapitanov/slovo")
        with ann_path.open("r", encoding="utf-8-sig", newline="") as fh:
            ann_rows = list(csv.DictReader(fh, delimiter="\t"))

        is_train = split == "train"
        rows = [
            r for r in ann_rows
            if str(r.get("train", "")).strip().lower() in (
                ("true", "1") if is_train else ("false", "0")
            )
        ]

        # ── class index ───────────────────────────────────────────────────
        all_classes = sorted({r["text"] for r in rows})
        rng = random.Random(seed)
        if max_classes is not None and max_classes < len(all_classes):
            all_classes = sorted(rng.sample(all_classes, max_classes))
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.rows = [r for r in rows if r["text"] in self.class_to_idx]

        # ── landmark source ───────────────────────────────────────────────
        if self._poses_dir.exists() and any(self._poses_dir.iterdir()):
            self._landmarks = None  # use per-sample .npy files
            print(f"{len(self.rows)} samples, {len(self.class_to_idx)} classes (npy mode).")
        else:
            # fallback: load full JSON (only feasible when RAM allows)
            lm_path = slovo_dir / "slovo_mediapipe.json"
            if not lm_path.exists():
                raise FileNotFoundError(
                    f"Missing {lm_path} and poses/ dir — run convert_slovo_to_npy.py first"
                )
            lm_key = str(lm_path.resolve())
            if lm_key not in _LANDMARK_CACHE:
                print(f"Loading {lm_path} (~1.2 GB)…", flush=True)
                with lm_path.open("r", encoding="utf-8") as fh:
                    _LANDMARK_CACHE[lm_key] = json.load(fh)
                print("Loaded.", flush=True)
            self._landmarks = _LANDMARK_CACHE[lm_key]
            print(f"{len(self.rows)} samples, {len(self.class_to_idx)} classes (json mode).")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from scripts.uzsl_v1.model import require_torch
        torch = require_torch()

        row = self.rows[idx]
        vid_id = row["attachment_id"]

        if self._landmarks is None:
            # .npy mode — load one file, zero-fill if missing
            npy_path = self._poses_dir / f"{vid_id}.npy"
            if npy_path.exists():
                arr = np.load(str(npy_path))   # (T, 75, 4)
            else:
                arr = np.zeros((0, _K, 4), np.float32)
        else:
            frames_json = self._landmarks.get(vid_id, [])
            arr = slovo_frames_to_array(frames_json) if frames_json else np.zeros((0, _K, 4), np.float32)

        if arr.shape[0] == 0:
            x = np.zeros((self.target_frames, _K * (10 if self.kinematics else 4)), np.float32)
            return torch.from_numpy(x), self.class_to_idx[row["text"]]

        arr = _wrist_normalize(arr)
        arr = trim_low_activity(arr, "hands_pose", threshold=0.01)
        if self.augment is not None:
            arr = self.augment(arr)
        arr = resample_frames(arr, self.target_frames)
        if self.kinematics:
            arr = add_kinematics(arr)
        x = arr.reshape(self.target_frames, -1)
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), self.class_to_idx[row["text"]]
