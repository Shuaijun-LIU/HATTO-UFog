from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2


def _try_get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return str(exe)
    except Exception:
        return None
    return None


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
    h = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h), interpolation=cv2.INTER_AREA)
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
    n = min(len(left_paths), len(right_paths))

    # Some runs may contain a few empty/corrupted frames or missing indices. To keep encoding robust
    # and high-quality, we first build a *contiguous* split-frame sequence, then encode it.
    tmp_dir = output_path.parent / "_tmp_split_frames"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    tmp_dir.mkdir(parents=True, exist_ok=True)

    first_frame = None
    frames_written = 0
    for i in range(n):
        try:
            left = _read_frame(left_paths[i])
            right = _read_frame(right_paths[i])
        except Exception:
            continue
        left, right = _ensure_same_height(left, right)
        frame = cv2.hconcat([left, right])
        if first_frame is None:
            first_frame = frame
        out_path = tmp_dir / f"frame_{frames_written:06d}.png"
        cv2.imwrite(str(out_path), frame)
        frames_written += 1

    if first_frame is None or frames_written <= 0:
        raise RuntimeError("Failed to find any readable left/right frame pairs for video encoding.")

    h = int(first_frame.shape[0])
    w = int(first_frame.shape[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _try_get_ffmpeg_exe()
    if ffmpeg:
        import subprocess

        # H.264 via ffmpeg (bundled via imageio-ffmpeg): substantially better quality than OpenCV mp4v.
        # Use CRF mode (visually lossless-ish) to avoid "blurry" textures on complex scenes.
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    else:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
        try:
            for i in range(frames_written):
                img = _read_frame(tmp_dir / f"frame_{i:06d}.png")
                writer.write(img)
        finally:
            writer.release()

    for p in tmp_dir.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return frames_written, w
