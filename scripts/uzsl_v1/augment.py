"""Pose-sequence augmentation for UzSL isolated SLR.

The stack follows the Kaggle GISLR 2023 winning recipes (temporal resample,
horizontal flip with left/right landmark swap, random affine, temporal and
landmark masking) plus the joint jitter from the sibling slr harness. With only
four signers, augmentation is the main substitute for signer diversity.

All transforms operate on full-length base arrays of shape (T, K, 4) with
channels (x, y, z, confidence) AFTER shoulder normalization (origin = shoulder
midpoint) and BEFORE resampling/kinematics. Conventions preserved:
  - undetected landmarks are all-zero with confidence 0 and stay that way
    (geometric transforms only touch visible landmarks),
  - the confidence channel is never jittered,
  - masked frames become all-zero, matching the "missing frame" convention.

Horizontal flip is only defined for the "hands_pose" component set (the face
mesh has no precomputed mirror permutation here).
"""
from __future__ import annotations

import numpy as np

# MediaPipe 33-point POSE_LANDMARKS left/right symmetric index pairs (nose=0 unpaired).
POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _hands_pose_flip_permutation() -> np.ndarray:
    """Landmark permutation mirroring the hands_pose layout [LH(21), POSE(33), RH(21)]."""
    perm = np.arange(75)
    perm[0:21], perm[54:75] = np.arange(54, 75), np.arange(0, 21)  # swap hands
    for a, b in POSE_LR_PAIRS:  # swap body left/right joints
        perm[21 + a], perm[21 + b] = 21 + b, 21 + a
    return perm


FLIP_PERMUTATIONS = {"hands_pose": _hands_pose_flip_permutation()}


class Augment:
    """Composable augmentation. Call on a (T, K, 4) float32 base array."""

    def __init__(
        self,
        *,
        components: str = "hands_pose",
        time_stretch_p: float = 0.8,
        time_stretch_range: tuple[float, float] = (0.7, 1.3),
        temporal_crop_p: float = 0.5,
        crop_min_frac: float = 0.7,
        flip_p: float = 0.5,
        affine_p: float = 0.75,
        rotate_deg: float = 15.0,
        scale_range: tuple[float, float] = (0.85, 1.15),
        shear: float = 0.1,
        shift: float = 0.08,
        jitter_std: float = 0.01,
        temporal_mask_p: float = 0.4,
        temporal_mask_frac: tuple[float, float] = (0.1, 0.3),
        landmark_mask_p: float = 0.3,
        landmark_mask_max: int = 6,
        body_proportion_p: float = 0.0,
        arm_scale_range: tuple[float, float] = (0.8, 1.2),
        seed: int | None = None,
    ) -> None:
        self.components = components
        self.time_stretch_p = time_stretch_p
        self.time_stretch_range = time_stretch_range
        self.temporal_crop_p = temporal_crop_p
        self.crop_min_frac = crop_min_frac
        self.flip_p = flip_p if components in FLIP_PERMUTATIONS else 0.0
        self.affine_p = affine_p
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range
        self.shear = shear
        self.shift = shift
        self.jitter_std = jitter_std
        self.temporal_mask_p = temporal_mask_p
        self.temporal_mask_frac = temporal_mask_frac
        self.landmark_mask_p = landmark_mask_p
        self.landmark_mask_max = landmark_mask_max
        # Only meaningful for hands_pose layout where shoulder indices are fixed
        self.body_proportion_p = body_proportion_p if components == "hands_pose" else 0.0
        self.arm_scale_range = arm_scale_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        x = np.array(arr, copy=True)
        if x.shape[0] == 0:
            return x

        # ── temporal ──────────────────────────────────────────────────────
        if x.shape[0] > 4 and self.rng.random() < self.time_stretch_p:
            factor = self.rng.uniform(*self.time_stretch_range)
            new_len = max(4, int(round(x.shape[0] * factor)))
            indices = np.round(np.linspace(0, x.shape[0] - 1, new_len)).astype(np.int64)
            x = x[indices]
        if x.shape[0] > 4 and self.rng.random() < self.temporal_crop_p:
            frac = self.rng.uniform(self.crop_min_frac, 1.0)
            window = max(4, int(round(x.shape[0] * frac)))
            if window < x.shape[0]:
                start = int(self.rng.integers(0, x.shape[0] - window + 1))
                x = x[start : start + window]

        visible = x[..., 3] > 0.0  # (T, K) — geometric ops touch these only

        # ── spatial ───────────────────────────────────────────────────────
        if self.flip_p > 0 and self.rng.random() < self.flip_p:
            x = x[:, FLIP_PERMUTATIONS[self.components]]
            x[..., 0] = -x[..., 0]
            visible = x[..., 3] > 0.0

        # Simulate different arm lengths / body proportions by scaling each hand
        # independently from its corresponding shoulder. Left shoulder = hands_pose
        # index 32 (pose[11]), right shoulder = 33 (pose[12]).
        if self.body_proportion_p > 0 and self.rng.random() < self.body_proportion_p:
            ls = x[:, 32, :3].copy()   # (T, 3) left shoulder position
            rs = x[:, 33, :3].copy()   # (T, 3) right shoulder position
            larm = float(self.rng.uniform(*self.arm_scale_range))
            rarm = float(self.rng.uniform(*self.arm_scale_range))
            lh_vis = (x[:, 0:21, 3] > 0)[..., np.newaxis]   # (T, 21, 1)
            rh_vis = (x[:, 54:75, 3] > 0)[..., np.newaxis]
            x[:, 0:21, :3] = np.where(
                lh_vis,
                ls[:, np.newaxis, :] + (x[:, 0:21, :3] - ls[:, np.newaxis, :]) * larm,
                x[:, 0:21, :3],
            )
            x[:, 54:75, :3] = np.where(
                rh_vis,
                rs[:, np.newaxis, :] + (x[:, 54:75, :3] - rs[:, np.newaxis, :]) * rarm,
                x[:, 54:75, :3],
            )

        if self.rng.random() < self.affine_p:
            angle = np.deg2rad(self.rng.uniform(-self.rotate_deg, self.rotate_deg))
            scale = self.rng.uniform(*self.scale_range)
            shear_x = self.rng.uniform(-self.shear, self.shear)
            cos, sin = np.cos(angle), np.sin(angle)
            matrix = scale * np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
            matrix = matrix @ np.array([[1.0, shear_x], [0.0, 1.0]], dtype=np.float32)
            offset = self.rng.uniform(-self.shift, self.shift, size=2).astype(np.float32)
            xy = x[..., :2][visible] @ matrix.T + offset
            x[..., :2][visible] = xy
            x[..., 2][visible] *= scale

        if self.jitter_std > 0:
            noise = self.rng.normal(0.0, self.jitter_std, size=(*x.shape[:2], 3)).astype(x.dtype)
            noise[~visible] = 0.0
            x[..., :3] += noise

        # ── masking ───────────────────────────────────────────────────────
        if x.shape[0] > 8 and self.rng.random() < self.temporal_mask_p:
            frac = self.rng.uniform(*self.temporal_mask_frac)
            span = max(1, int(round(x.shape[0] * frac)))
            start = int(self.rng.integers(0, x.shape[0] - span + 1))
            x[start : start + span] = 0.0

        if self.landmark_mask_max > 0 and self.rng.random() < self.landmark_mask_p:
            count = int(self.rng.integers(1, self.landmark_mask_max + 1))
            landmarks = self.rng.choice(x.shape[1], size=count, replace=False)
            x[:, landmarks] = 0.0

        return x
