#!/usr/bin/env python3
"""Sidecar extractor: runs MediaPipe Holistic (model_complexity=2) on a video
and writes a .pose file. Called as a subprocess from the webapp so that it
runs in the pose_venv310 environment which has mediapipe==0.10.14.

Usage:
    pose_venv310/Scripts/python.exe scripts/uzsl_v1/pose_extract_holistic.py <video> <pose>
"""
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: pose_extract_holistic.py <video_path> <pose_path>")

    video_path = sys.argv[1]
    pose_path = sys.argv[2]

    try:
        from pose_format.bin.pose_estimation import pose_video
    except ImportError as exc:
        sys.exit(
            f"pose_format not available: {exc}\n"
            "Run with pose_venv310 which has mediapipe==0.10.14 and pose-format installed."
        )

    try:
        pose_video(
            video_path,
            pose_path,
            "mediapipe",
            {"model_complexity": 2},
            progress=False,
        )
    except Exception as exc:
        sys.exit(f"Extraction failed: {exc}")


if __name__ == "__main__":
    main()
