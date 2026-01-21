from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2


def _sorted_pngs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png"))


def _read_frame(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _ensure_same_height(left, right):
    if left.shape[0] == right.shape[0]:
        return left, right
    height_px = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * height_px / left.shape[0]), height_px), interpolation=cv2.INTER_AREA)
    right = cv2.resize(
        right, (int(right.shape[1] * height_px / right.shape[0]), height_px), interpolation=cv2.INTER_AREA
    )
    return left, right


def build_split_video(
    *,
    frames_left_dir: Path,
    frames_right_dir: Path,
    output_path: Path,
    fps: float,
) -> Tuple[int, int]:
    """Create one MP4 with left/right views concatenated.

    Returns: (num_frames_written, width_px)
    """

    left_paths = _sorted_pngs(frames_left_dir)
    right_paths = _sorted_pngs(frames_right_dir)
    if not left_paths or not right_paths:
        raise RuntimeError("No frames found (need both left/right PNG sequences).")
    num_frames = min(len(left_paths), len(right_paths))

    first_left = _read_frame(left_paths[0])
    first_right = _read_frame(right_paths[0])
    first_left, first_right = _ensure_same_height(first_left, first_right)
    height_px = first_left.shape[0]
    width_px = first_left.shape[1] + first_right.shape[1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width_px, height_px),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    try:
        for i in range(num_frames):
            left = _read_frame(left_paths[i])
            right = _read_frame(right_paths[i])
            left, right = _ensure_same_height(left, right)
            frame = cv2.hconcat([left, right])
            writer.write(frame)
    finally:
        writer.release()
    return num_frames, width_px

