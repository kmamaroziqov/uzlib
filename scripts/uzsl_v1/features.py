"""Vectorized v2 feature pipeline for UzSL isolated SLR.

Replaces the per-landmark Python loops in pose_io with numpy operations and the
per-frame bbox normalization with the field-standard per-clip shoulder-span
normalization (the recipe used by upstream sign-language-processing/recognition
and the AUTSL/GISLR systems). The confidence channel is kept: it explicitly
encodes "landmark not detected", which plain zero-filling conflates with
"landmark at the origin".

Pipeline:
    load_feature_array  -> (T, K, 4) float32 in image-normalized coordinates
    shoulder_normalize  -> per-clip center on shoulder midpoint, scale by span
    trim_low_activity   -> drop dead lead-in/lead-out frames
    resample_frames     -> uniform resample to a fixed frame count
    add_kinematics      -> append per-landmark velocity + acceleration (xyz)
    flatten_frames      -> (T, K * C) model input

`prepare_base` bundles the cacheable prefix (load + normalize + trim), so a
training dataset can keep full-length base arrays in memory and apply
augmentation + resampling + kinematics per access.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .pose_io import COMPONENT_COUNTS, FEATURE_COMPONENTS

LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
VISIBLE_CONF = 0.05
BASE_CHANNELS = 4  # x, y, z, confidence
KINEMATIC_CHANNELS = 10  # + velocity xyz + acceleration xyz


def component_layout(components: str) -> list[tuple[str, int, int]]:
    layout: list[tuple[str, int, int]] = []
    offset = 0
    for name in FEATURE_COMPONENTS[components]:
        count = COMPONENT_COUNTS[name]
        layout.append((name, offset, offset + count))
        offset += count
    return layout


def component_slice(components: str, name: str) -> slice:
    for layout_name, start, end in component_layout(components):
        if layout_name == name:
            return slice(start, end)
    raise KeyError(f"Component {name!r} not in feature set {components!r}")


def landmark_count(components: str) -> int:
    return sum(COMPONENT_COUNTS[name] for name in FEATURE_COMPONENTS[components])


def feature_dim(components: str, *, kinematics: bool = False) -> int:
    channels = KINEMATIC_CHANNELS if kinematics else BASE_CHANNELS
    return landmark_count(components) * channels


def load_feature_array(path: Path, components: str = "hands_pose") -> np.ndarray:
    """Read a .pose file into a (T, K, 4) float32 array without Python loops.

    x/y are returned in image-normalized [0, 1] space: files written by this
    repo's extractor already store normalized MediaPipe coordinates, while
    upstream-extracted files store pixel coordinates and are detected by
    magnitude and divided by the header (width, height).
    """
    try:
        from pose_format import Pose
    except ImportError as exc:
        raise SystemExit(
            "Reading .pose files requires pose-format. "
            "Install dependencies with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    with Path(path).open("rb") as fh:
        pose = Pose.read(fh.read())

    data = np.asarray(pose.body.data.filled(0), dtype=np.float32)[:, 0]  # (T, K_file, 3)
    conf = np.asarray(pose.body.confidence, dtype=np.float32)[:, 0]  # (T, K_file)
    frame_count = data.shape[0]
    if frame_count == 0:
        raise ValueError(f"{path}: pose file has no frames")

    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        offsets[component.name] = (offset, offset + count)
        offset += count

    parts: list[np.ndarray] = []
    for name in FEATURE_COMPONENTS[components]:
        expected = COMPONENT_COUNTS[name]
        block = np.zeros((frame_count, expected, BASE_CHANNELS), dtype=np.float32)
        if name in offsets:
            start, end = offsets[name]
            available = min(expected, end - start)
            block[:, :available, :3] = data[:, start : start + available]
            block[:, :available, 3] = conf[:, start : start + available]
        parts.append(block)
    arr = np.concatenate(parts, axis=1)
    arr[~np.isfinite(arr)] = 0.0

    # Upstream-extracted files store pixel coordinates; ours store [0, 1].
    if np.abs(arr[..., :2]).max() > 2.0:
        width = float(pose.header.dimensions.width) or 1.0
        height = float(pose.header.dimensions.height) or 1.0
        arr[..., 0] /= width
        arr[..., 1] /= height
    return arr


def shoulder_normalize(arr: np.ndarray, components: str = "hands_pose") -> np.ndarray:
    """Per-clip normalization: center on the shoulder midpoint, scale by span.

    Statistics are computed once per clip over frames where both shoulders are
    visible, so frame-to-frame visibility flicker cannot inject jitter the way
    per-frame bbox normalization does. Undetected landmarks (confidence 0) are
    re-zeroed afterwards so "missing" stays distinguishable from a position.
    """
    pose_block = component_slice(components, "POSE_LANDMARKS")
    left = arr[:, pose_block.start + LEFT_SHOULDER_IDX]
    right = arr[:, pose_block.start + RIGHT_SHOULDER_IDX]
    visible = (left[:, 3] > VISIBLE_CONF) & (right[:, 3] > VISIBLE_CONF)

    if visible.any():
        mid = (left[visible, :3] + right[visible, :3]) / 2.0
        center = mid.mean(axis=0)
        span = np.linalg.norm(left[visible, :2] - right[visible, :2], axis=1).mean()
        scale = max(float(span), 1e-4)
    else:
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 1.0

    out = arr.copy()
    out[..., :3] = (out[..., :3] - center) / scale
    out[..., :3][out[..., 3] <= 0.0] = 0.0
    return out


def wrist_normalize_hands(arr: np.ndarray, components: str = "hands_pose") -> np.ndarray:
    """Per-hand normalization: center on wrist, scale by hand span.

    Removes arm-length bias while keeping wrist position intact inside
    POSE_LANDMARKS (indices 15/16), so the model still knows WHERE the
    hands are relative to the body.
    """
    out = arr.copy()
    for comp_name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
        try:
            sl = component_slice(components, comp_name)
        except KeyError:
            continue
        wrist_idx = sl.start  # WRIST is the first landmark in MediaPipe hand model
        for t in range(arr.shape[0]):
            if out[t, wrist_idx, 3] <= VISIBLE_CONF:
                continue
            wrist_pos = out[t, wrist_idx, :3].copy()
            vis = out[t, sl.start:sl.stop, 3] > VISIBLE_CONF  # (21,)
            if vis.sum() < 2:
                continue
            out[t, sl.start:sl.stop, :3] -= wrist_pos
            pts = out[t, sl.start:sl.stop, :3][vis]
            diffs = pts[:, np.newaxis] - pts[np.newaxis]
            span = float(np.sqrt((diffs ** 2).sum(-1)).max())
            if span > 1e-4:
                out[t, sl.start:sl.stop, :3][vis] /= span
            out[t, sl.start:sl.stop, :3][~vis] = 0.0
    return out


def trim_low_activity(
    arr: np.ndarray,
    components: str = "hands_pose",
    threshold: float = 0.03,
) -> np.ndarray:
    """Drop dead lead-in/lead-out frames (hands down, nothing detected)."""
    if threshold <= 0 or arr.shape[0] == 0:
        return arr
    left = component_slice(components, "LEFT_HAND_LANDMARKS")
    right = component_slice(components, "RIGHT_HAND_LANDMARKS")
    pose = component_slice(components, "POSE_LANDMARKS")
    hand_conf = np.concatenate([arr[:, left, 3], arr[:, right, 3]], axis=1)
    pose_conf = arr[:, pose, 3]
    activity = np.where(hand_conf.max(axis=1) > 0, hand_conf.mean(axis=1), pose_conf.mean(axis=1))
    active = np.flatnonzero(activity >= threshold)
    if active.size == 0:
        return arr
    return arr[active[0] : active[-1] + 1]


def resample_frames(arr: np.ndarray, target_frames: int) -> np.ndarray:
    """Uniformly resample to exactly `target_frames` (never truncates content)."""
    length = arr.shape[0]
    if length == target_frames:
        return arr
    if length <= 1:
        return np.repeat(arr[:1] if length else np.zeros((1, *arr.shape[1:]), arr.dtype), target_frames, axis=0)
    indices = np.round(np.linspace(0, length - 1, target_frames)).astype(np.int64)
    return arr[indices]


def add_kinematics(arr: np.ndarray) -> np.ndarray:
    """Append xyz velocity and acceleration channels: (T, K, 4) -> (T, K, 10)."""
    xyz = arr[..., :3]
    velocity = np.diff(xyz, axis=0, prepend=xyz[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    return np.concatenate([arr, velocity, acceleration], axis=-1)


def flatten_frames(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.reshape(arr.shape[0], -1), dtype=np.float32)


def prepare_base(
    path: Path,
    components: str = "hands_pose",
    *,
    trim_threshold: float = 0.03,
    wrist_norm: bool = False,
) -> np.ndarray:
    """The cacheable prefix: load + normalize + trim, kept at full length."""
    arr = load_feature_array(path, components)
    arr = shoulder_normalize(arr, components)
    if wrist_norm:
        arr = wrist_normalize_hands(arr, components)
    return trim_low_activity(arr, components, trim_threshold)


def finalize(
    arr: np.ndarray,
    target_frames: int,
    *,
    kinematics: bool = False,
) -> np.ndarray:
    """Resample, optionally add kinematics, and flatten to (T, K * C)."""
    arr = resample_frames(arr, target_frames)
    if kinematics:
        arr = add_kinematics(arr)
    return flatten_frames(arr)
