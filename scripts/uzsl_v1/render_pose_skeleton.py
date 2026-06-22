#!/usr/bin/env python3
"""Sidecar renderer: renders a .pose file to a skeleton MP4.
Run with pose_venv310 which has mediapipe==0.10.14 and pose-format installed.

Usage:
    pose_venv310/Scripts/python.exe scripts/uzsl_v1/render_pose_skeleton.py <pose_path> <mp4_path>
"""
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: render_pose_skeleton.py <pose_path> <mp4_path>")

    pose_path, mp4_path = sys.argv[1], sys.argv[2]

    try:
        from pose_format import Pose
        from pose_format.pose_visualizer import PoseVisualizer
    except ImportError as exc:
        sys.exit(f"pose_format not available: {exc}")

    try:
        import cv2
    except ImportError as exc:
        sys.exit(f"opencv not available: {exc}")

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    # Keep only the landmarks used for signing (drop noisy face interior)
    try:
        pose = pose.get_components(
            ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
        )
    except Exception:
        pass  # fall back to all components if filtering fails

    W = pose.header.dimensions.width or 640
    H = pose.header.dimensions.height or 480
    fps = pose.body.fps or 25

    v = PoseVisualizer(pose, thickness=2)
    frames = list(v.draw(background_color=(20, 20, 20)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    main()
