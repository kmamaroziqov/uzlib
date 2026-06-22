from __future__ import annotations

import argparse
import os
import csv
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .paths import DEFAULT_DATA_DIR, DEFAULT_MANIFEST
from .pose_io import (
    LANDMARK_COUNT,
    landmark_list_to_values,
    landmarks_to_values,
    missing_landmarks,
    write_pose_file,
)
from .progress import ProgressBar


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "generated" / "mediapipe_models"


def configure_native_logging() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def download_model(path: Path, url: str, *, allow_download: bool) -> None:
    if path.exists():
        return
    if not allow_download:
        raise RuntimeError(f"Missing MediaPipe model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def task_hand_values(hand_result) -> tuple[list[list[float]], list[list[float]]]:
    left = missing_landmarks(21)
    right = missing_landmarks(21)
    handedness = getattr(hand_result, "handedness", []) or []
    hand_landmarks = getattr(hand_result, "hand_landmarks", []) or []
    for marks, handed in zip(hand_landmarks, handedness):
        label = ""
        if handed:
            label = getattr(handed[0], "category_name", "") or getattr(handed[0], "display_name", "")
        values = landmarks_to_values(marks, 21)
        if label.lower() == "left":
            left = values
        elif label.lower() == "right":
            right = values
    return left, right


def extract_video_with_tasks(
    video_path: Path,
    pose_path: Path,
    *,
    model_dir: Path,
    allow_model_download: bool,
    delegate: str = "cpu",
    max_frames: int | None = None,
    model_quality: str = "full",
) -> int:
    configure_native_logging()
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
    except ImportError as exc:
        raise SystemExit(
            "Pose extraction requires opencv-python and mediapipe. "
            "Install them with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    pose_model_name = f"pose_landmarker_{model_quality}.task"
    pose_model = model_dir / pose_model_name
    hand_model = model_dir / "hand_landmarker.task"
    face_model = model_dir / "face_landmarker.task"
    pose_url = POSE_MODEL_URL.replace("pose_landmarker_lite", f"pose_landmarker_{model_quality}")
    download_model(pose_model, pose_url, allow_download=allow_model_download)
    download_model(hand_model, HAND_MODEL_URL, allow_download=allow_model_download)
    download_model(face_model, FACE_MODEL_URL, allow_download=allow_model_download)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    component_frames: dict[str, list[list[list[float]]]] = {
        "POSE_LANDMARKS": [],
        "FACE_LANDMARKS": [],
        "LEFT_HAND_LANDMARKS": [],
        "RIGHT_HAND_LANDMARKS": [],
        "POSE_WORLD_LANDMARKS": [],
    }

    delegate_value = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model), delegate=delegate_value),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model), delegate=delegate_value),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
    )
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_model), delegate=delegate_value),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
    )

    try:
        with (
            PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
            HandLandmarker.create_from_options(hand_options) as hand_landmarker,
            FaceLandmarker.create_from_options(face_options) as face_landmarker,
        ):
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_index * 1000 / fps)
                pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)
                face_result = face_landmarker.detect_for_video(image, timestamp_ms)

                pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                pose_world_landmarks = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None
                face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
                left_hand, right_hand = task_hand_values(hand_result)
                component_frames["POSE_LANDMARKS"].append(landmarks_to_values(pose_landmarks, 33))
                component_frames["FACE_LANDMARKS"].append(landmarks_to_values(face_landmarks, 468))
                component_frames["LEFT_HAND_LANDMARKS"].append(left_hand)
                component_frames["RIGHT_HAND_LANDMARKS"].append(right_hand)
                component_frames["POSE_WORLD_LANDMARKS"].append(landmarks_to_values(pose_world_landmarks, 33))
                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
    finally:
        cap.release()

    if not component_frames["POSE_LANDMARKS"]:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    write_pose_file(
        pose_path,
        source_video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        component_frames=component_frames,
    )
    return len(component_frames["POSE_LANDMARKS"])


def extract_video_with_legacy_solutions(video_path: Path, pose_path: Path, *, max_frames: int | None = None) -> int:
    configure_native_logging()
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "Pose extraction requires opencv-python and mediapipe. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[list[list[float]]] = []

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            values = []
            values.extend(landmark_list_to_values(result.pose_landmarks, 33))
            values.extend(landmark_list_to_values(result.left_hand_landmarks, 21))
            values.extend(landmark_list_to_values(result.right_hand_landmarks, 21))
            if len(values) != LANDMARK_COUNT:
                raise RuntimeError(f"Internal landmark count mismatch: {len(values)}")
            frames.append(values)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        holistic.close()
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    write_pose_file(pose_path, source_video=str(video_path), fps=fps, width=width, height=height, frames=frames)
    return len(frames)


def extract_video(
    video_path: Path,
    pose_path: Path,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    allow_model_download: bool = True,
    delegate: str = "cpu",
    max_frames: int | None = None,
    model_quality: str = "full",
) -> int:
    configure_native_logging()
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "Pose extraction requires opencv-python and mediapipe. "
            "Install them with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    if hasattr(mp, "solutions"):
        return extract_video_with_legacy_solutions(video_path, pose_path, max_frames=max_frames)
    return extract_video_with_tasks(
        video_path,
        pose_path,
        model_dir=model_dir,
        allow_model_download=allow_model_download,
        delegate=delegate,
        max_frames=max_frames,
        model_quality=model_quality,
    )


def default_worker_count(delegate: str) -> int:
    if delegate == "gpu":
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def resolve_output_paths(row: dict[str, str], data_dir: Path) -> tuple[Path, Path]:
    video_path = data_dir / row["video_path"]
    pose_path = Path(row["pose_path"])
    if not pose_path.is_absolute():
        if pose_path.parts and pose_path.parts[0] == data_dir.name:
            pose_path = data_dir.parent / pose_path
        else:
            pose_path = data_dir / pose_path
    return video_path, pose_path


def extract_one_job(args: tuple[dict[str, str], str, bool, str, bool, str, int | None, str]) -> dict[str, object]:
    configure_native_logging()
    row, data_dir_s, overwrite, model_dir_s, allow_model_download, delegate, max_frames, model_quality = args
    data_dir = Path(data_dir_s)
    video_path, pose_path = resolve_output_paths(row, data_dir)
    if pose_path.exists() and not overwrite:
        return {"status": "skipped", "sample_id": row["sample_id"], "pose_path": str(pose_path), "frames": 0}
    try:
        frame_count = extract_video(
            video_path,
            pose_path,
            model_dir=Path(model_dir_s),
            allow_model_download=allow_model_download,
            delegate=delegate,
            max_frames=max_frames,
            model_quality=model_quality,
        )
        return {
            "status": "extracted",
            "sample_id": row["sample_id"],
            "pose_path": str(pose_path),
            "frames": frame_count,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "sample_id": row.get("sample_id", ""),
            "video_path": str(video_path),
            "error": str(exc),
        }


def extract_from_manifest(
    manifest_path: Path,
    data_dir: Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    model_dir: Path = DEFAULT_MODEL_DIR,
    allow_model_download: bool = True,
    workers: int | None = None,
    delegate: str = "cpu",
    progress: bool = True,
    max_frames: int | None = None,
    model_quality: str = "full",
) -> dict[str, int]:
    rows = read_manifest(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    workers = default_worker_count(delegate) if workers is None else max(1, workers)
    if delegate == "gpu" and workers > 1:
        print("warning: --delegate gpu is usually fastest and most stable with --workers 1")
    if allow_model_download:
        pose_url = POSE_MODEL_URL.replace("pose_landmarker_lite", f"pose_landmarker_{model_quality}")
        download_model(model_dir / f"pose_landmarker_{model_quality}.task", pose_url, allow_download=True)
        download_model(model_dir / "hand_landmarker.task", HAND_MODEL_URL, allow_download=True)
        download_model(model_dir / "face_landmarker.task", FACE_MODEL_URL, allow_download=True)

    extracted = 0
    skipped = 0
    failed = 0
    processed = len(rows)
    bar = ProgressBar(processed, label="poses") if progress else None
    done = 0
    failures: list[str] = []

    jobs = [
        (row, str(data_dir), overwrite, str(model_dir), allow_model_download, delegate, max_frames, model_quality)
        for row in rows
    ]
    if workers == 1:
        iterator = (extract_one_job(job) for job in jobs)
        for result in iterator:
            status = str(result["status"])
            extracted += int(status == "extracted")
            skipped += int(status == "skipped")
            failed += int(status == "failed")
            if status == "failed":
                failures.append(f"{result.get('sample_id')}: {result.get('error')}")
            done += 1
            if bar:
                bar.update(done, suffix=f"ok {extracted}, skipped {skipped}, failed {failed}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_one_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                status = str(result["status"])
                extracted += int(status == "extracted")
                skipped += int(status == "skipped")
                failed += int(status == "failed")
                if status == "failed":
                    failures.append(f"{result.get('sample_id')}: {result.get('error')}")
                done += 1
                if bar:
                    bar.update(done, suffix=f"ok {extracted}, skipped {skipped}, failed {failed}")

    if bar:
        bar.finish(suffix=f"ok {extracted}, skipped {skipped}, failed {failed}")
    for failure in failures[:25]:
        print(f"failed {failure}")
    if len(failures) > 25:
        print(f"... {len(failures) - 25} more failures")
    return {"processed": processed, "extracted": extracted, "skipped": skipped, "failed": failed}


def main() -> None:
    configure_native_logging()
    parser = argparse.ArgumentParser(description="Extract generated MediaPipe .pose files from UzSL videos.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--no-download-models", action="store_true")
    parser.add_argument("--workers", default="auto", help="Number of parallel videos to process, or 'auto'.")
    parser.add_argument("--delegate", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model-quality", choices=["lite", "full", "heavy"], default="full")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    workers = None if args.workers == "auto" else int(args.workers)

    stats = extract_from_manifest(
        args.manifest,
        args.data_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        model_dir=args.model_dir,
        allow_model_download=not args.no_download_models,
        workers=workers,
        delegate=args.delegate,
        progress=not args.no_progress,
        max_frames=args.max_frames,
        model_quality=args.model_quality,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
