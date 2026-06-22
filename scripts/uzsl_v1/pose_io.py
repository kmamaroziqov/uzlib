from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


POSE_LANDMARKS = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]
LEFT_HAND_LANDMARKS = HAND_LANDMARKS
RIGHT_HAND_LANDMARKS = HAND_LANDMARKS
LANDMARK_NAMES = POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS
LANDMARK_COUNT = len(LANDMARK_NAMES)
VALUES_PER_LANDMARK = 4
POSE_COMPONENT_NAMES = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
FACE_LANDMARKS = [str(i) for i in range(468)]
POSE_WORLD_LANDMARKS = POSE_LANDMARKS
COMPONENT_SPECS = [
    ("POSE_LANDMARKS", POSE_LANDMARKS, (255, 0, 0)),
    ("FACE_LANDMARKS", FACE_LANDMARKS, (255, 255, 255)),
    ("LEFT_HAND_LANDMARKS", LEFT_HAND_LANDMARKS, (0, 255, 0)),
    ("RIGHT_HAND_LANDMARKS", RIGHT_HAND_LANDMARKS, (0, 0, 255)),
    ("POSE_WORLD_LANDMARKS", POSE_WORLD_LANDMARKS, (255, 128, 0)),
]
COMPONENT_COUNTS = {name: len(points) for name, points, _ in COMPONENT_SPECS}
FEATURE_COMPONENTS = {
    "hands_pose": ["LEFT_HAND_LANDMARKS", "POSE_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
    "rec": ["FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "POSE_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
    "full": ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS", "POSE_WORLD_LANDMARKS"],
}


def missing_landmarks(count: int) -> list[list[float]]:
    return [[0.0, 0.0, 0.0, 0.0] for _ in range(count)]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    mask = getattr(value, "mask", False)
    try:
        if bool(mask):
            return default
    except ValueError:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def landmark_list_to_values(landmark_list: Any, count: int) -> list[list[float]]:
    if landmark_list is None:
        return missing_landmarks(count)
    values: list[list[float]] = []
    for landmark in landmark_list.landmark:
        confidence = getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))
        values.append(
            [
                safe_float(getattr(landmark, "x", None)),
                safe_float(getattr(landmark, "y", None)),
                safe_float(getattr(landmark, "z", None)),
                safe_float(confidence),
            ]
        )
    if len(values) < count:
        values.extend(missing_landmarks(count - len(values)))
    return values[:count]


def landmarks_to_values(landmarks: Any, count: int) -> list[list[float]]:
    if landmarks is None:
        return missing_landmarks(count)
    values: list[list[float]] = []
    for landmark in landmarks:
        confidence = getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))
        values.append(
            [
                safe_float(getattr(landmark, "x", None)),
                safe_float(getattr(landmark, "y", None)),
                safe_float(getattr(landmark, "z", None)),
                safe_float(confidence),
            ]
        )
    if len(values) < count:
        values.extend(missing_landmarks(count - len(values)))
    return values[:count]


def write_pose_json(
    path: Path,
    *,
    source_video: str,
    fps: float,
    width: int,
    height: int,
    frames: list[list[list[float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "schema": "mediapipe_pose_hands_v1",
        "source_video": source_video,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "landmark_names": LANDMARK_NAMES,
        "values": "x,y,z,confidence",
        "frames": frames,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def write_pose_file(
    path: Path,
    *,
    source_video: str,
    fps: float,
    width: int,
    height: int,
    frames: list[list[list[float]]] | None = None,
    component_frames: dict[str, list[list[list[float]]]] | None = None,
) -> None:
    try:
        import numpy as np
        from pose_format import Pose
        from pose_format.numpy.pose_body import NumPyPoseBody
        from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
    except ImportError as exc:
        raise SystemExit(
            "Writing .pose files requires pose-format. "
            "Install dependencies with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if component_frames is None:
        if frames is None:
            raise ValueError("write_pose_file needs frames or component_frames")
        component_frames = {
            "POSE_LANDMARKS": [frame[:33] for frame in frames],
            "LEFT_HAND_LANDMARKS": [frame[33:54] for frame in frames],
            "RIGHT_HAND_LANDMARKS": [frame[54:75] for frame in frames],
        }
    active_specs = [(name, points, color) for name, points, color in COMPONENT_SPECS if name in component_frames]
    components = [PoseHeaderComponent(name, points, [], [color], "XYZC") for name, points, color in active_specs]
    header = PoseHeader(
        version=0.1,
        dimensions=PoseHeaderDimensions(width=int(width), height=int(height), depth=0),
        components=components,
    )
    first_component = next(iter(component_frames.values()))
    frame_count = len(first_component)
    merged_frames: list[list[list[float]]] = []
    for frame_index in range(frame_count):
        merged: list[list[float]] = []
        for name, points, _ in active_specs:
            values = component_frames[name][frame_index]
            expected_count = len(points)
            merged.extend(values[:expected_count])
            if len(values) < expected_count:
                merged.extend(missing_landmarks(expected_count - len(values)))
        merged_frames.append(merged)
    coords = np.array([[[[point[0], point[1], point[2]] for point in frame]] for frame in merged_frames], dtype=np.float32)
    confidence = np.array([[[point[3] for point in frame]] for frame in merged_frames], dtype=np.float32)
    pose = Pose(header, NumPyPoseBody(fps=float(fps or 0.0), data=coords, confidence=confidence))
    with path.open("wb") as fh:
        pose.write(fh)


def read_pose_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    validate_pose_payload(payload, path)
    return payload


def read_pose_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return read_pose_json(path)
    try:
        from pose_format import Pose
    except ImportError as exc:
        raise SystemExit(
            "Reading .pose files requires pose-format. "
            "Install dependencies with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    with path.open("rb") as fh:
        pose = Pose.read(fh.read())
    component_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for component in pose.header.components:
        count = len(component.points)
        component_offsets[component.name] = (offset, offset + count)
        offset += count

    component_frames: dict[str, list[list[list[float]]]] = {name: [] for name, _, _ in COMPONENT_SPECS}
    data = pose.body.data
    confidence = pose.body.confidence
    for frame_index in range(data.shape[0]):
        for component_name, expected_count in COMPONENT_COUNTS.items():
            start_end = component_offsets.get(component_name)
            if start_end is None:
                component_frames[component_name].append(missing_landmarks(expected_count))
                continue
            start, end = start_end
            points = data[frame_index, 0, start:end, :]
            conf = confidence[frame_index, 0, start:end]
            values: list[list[float]] = []
            for point, point_conf in zip(points[:expected_count], conf[:expected_count]):
                values.append([safe_float(point[0]), safe_float(point[1]), safe_float(point[2]), safe_float(point_conf)])
            if len(points) < expected_count:
                values.extend(missing_landmarks(expected_count - len(points)))
            component_frames[component_name].append(values)

    frames = flatten_components(component_frames, FEATURE_COMPONENTS["hands_pose"])

    payload = {
        "version": 1,
        "schema": "mediapipe_pose_hands_v1",
        "source_video": "",
        "fps": float(pose.body.fps),
        "width": int(pose.header.dimensions.width),
        "height": int(pose.header.dimensions.height),
        "frame_count": len(frames),
        "landmark_names": LANDMARK_NAMES,
        "values": "x,y,z,confidence",
        "component_frames": component_frames,
        "frames": frames,
    }
    validate_pose_payload(payload, path)
    return payload


def flatten_components(payload_or_components: dict[str, Any], components: list[str]) -> list[list[list[float]]]:
    component_frames = payload_or_components.get("component_frames", payload_or_components)
    available = [component_frames[name] for name in components if name in component_frames]
    if not available:
        return []
    frame_count = len(available[0])
    frames: list[list[list[float]]] = []
    for frame_index in range(frame_count):
        frame: list[list[float]] = []
        for name in components:
            values = component_frames.get(name)
            if values is None:
                frame.extend(missing_landmarks(COMPONENT_COUNTS[name]))
            else:
                frame.extend(values[frame_index])
        frames.append(frame)
    return frames


def validate_pose_payload(payload: dict[str, Any], path: Path | None = None) -> None:
    prefix = f"{path}: " if path else ""
    if payload.get("schema") != "mediapipe_pose_hands_v1":
        raise ValueError(f"{prefix}unsupported pose schema: {payload.get('schema')!r}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{prefix}pose file has no frames")
    expected_names = payload.get("landmark_names", [])
    if len(expected_names) != LANDMARK_COUNT:
        raise ValueError(f"{prefix}expected {LANDMARK_COUNT} landmarks, got {len(expected_names)}")
    for frame_index, frame in enumerate(frames[:3]):
        if len(frame) != LANDMARK_COUNT:
            raise ValueError(f"{prefix}frame {frame_index} has {len(frame)} landmarks")
        for landmark in frame[:3]:
            if len(landmark) != VALUES_PER_LANDMARK:
                raise ValueError(f"{prefix}landmark should have 4 values")


def resample_indices(length: int, target_length: int) -> list[int]:
    if length <= 0:
        return [0] * target_length
    if target_length <= 1:
        return [0]
    return [round(i * (length - 1) / (target_length - 1)) for i in range(target_length)]


def frame_activity(frame: list[list[float]]) -> float:
    hand_points = frame[33:]
    hand_conf = [point[3] for point in hand_points]
    if any(conf > 0 for conf in hand_conf):
        return sum(hand_conf) / len(hand_conf)
    pose_conf = [point[3] for point in frame[:33]]
    return sum(pose_conf) / len(pose_conf)


def trim_meaningless_frames(frames: list[list[list[float]]], threshold: float = 0.03) -> list[list[list[float]]]:
    active = [idx for idx, frame in enumerate(frames) if frame_activity(frame) >= threshold]
    if not active:
        return frames
    return frames[min(active) : max(active) + 1]


def normalized_flat_features(
    payload: dict[str, Any],
    target_frames: int = 64,
    *,
    trim_threshold: float = 0.03,
    components: str = "hands_pose",
) -> list[float]:
    frames = flatten_components(payload, FEATURE_COMPONENTS[components]) if "component_frames" in payload else payload["frames"]
    pose_frames = flatten_components(payload, ["POSE_LANDMARKS"]) if "component_frames" in payload else [frame[:33] for frame in frames]
    if trim_threshold > 0:
        frames = trim_meaningless_frames(frames, trim_threshold)
    indices = resample_indices(len(frames), target_frames)
    result: list[float] = []
    for idx in indices:
        frame = frames[idx]
        pose = pose_frames[idx] if idx < len(pose_frames) else frame[:33]
        visible_pose = [p for p in pose if p[3] > 0.05]
        if visible_pose:
            cx = sum(p[0] for p in visible_pose) / len(visible_pose)
            cy = sum(p[1] for p in visible_pose) / len(visible_pose)
            xs = [p[0] for p in visible_pose]
            ys = [p[1] for p in visible_pose]
            scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-4)
        else:
            cx, cy, scale = 0.5, 0.5, 1.0
        for x, y, z, conf in frame:
            result.extend([(x - cx) / scale, (y - cy) / scale, z / scale, conf])
    return result


def normalized_sequence_features(
    payload: dict[str, Any],
    target_frames: int = 64,
    *,
    trim_threshold: float = 0.03,
    components: str = "hands_pose",
) -> list[list[float]]:
    frames = flatten_components(payload, FEATURE_COMPONENTS[components]) if "component_frames" in payload else payload["frames"]
    pose_frames = flatten_components(payload, ["POSE_LANDMARKS"]) if "component_frames" in payload else [frame[:33] for frame in frames]
    if trim_threshold > 0:
        frames = trim_meaningless_frames(frames, trim_threshold)
    indices = resample_indices(len(frames), target_frames)
    result: list[list[float]] = []
    for idx in indices:
        frame = frames[idx]
        pose = pose_frames[idx] if idx < len(pose_frames) else frame[:33]
        visible = [p for p in pose if p[3] > 0.05]
        if visible:
            cx = sum(p[0] for p in visible) / len(visible)
            cy = sum(p[1] for p in visible) / len(visible)
            xs = [p[0] for p in visible]
            ys = [p[1] for p in visible]
            scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-4)
        else:
            cx, cy, scale = 0.5, 0.5, 1.0
        frame_features: list[float] = []
        for x, y, z, conf in frame:
            frame_features.extend([(x - cx) / scale, (y - cy) / scale, z / scale, conf])
        result.append(frame_features)
    return result


def component_feature_dim(components: str) -> int:
    return sum(COMPONENT_COUNTS[name] for name in FEATURE_COMPONENTS[components]) * 4
