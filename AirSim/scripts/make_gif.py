from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio


def _resize_rgb(rgb, width: int | None) -> "list[list[list[int]]] | object":
    if width is None or width <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if w <= 0 or h <= 0:
        return rgb
    if w == width:
        return rgb
    scale = float(width) / float(w)
    new_h = max(1, int(round(float(h) * scale)))
    out = cv2.resize(rgb, (int(width), int(new_h)), interpolation=cv2.INTER_AREA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a GIF preview from an MP4 without requiring system ffmpeg.")
    ap.add_argument("--input", required=True, help="Input MP4 path")
    ap.add_argument("--output", required=True, help="Output GIF path")
    ap.add_argument("--width", type=int, default=960, help="Output GIF width (keeps aspect). Use 0 to keep original.")
    ap.add_argument("--fps", type=float, default=12.0, help="Target GIF fps (sampling from video).")
    ap.add_argument("--max_s", type=float, default=10.0, help="Max seconds to include from the start of the video.")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps_in <= 1e-6:
        fps_in = 20.0
    fps_out = max(1.0, float(args.fps))
    stride = max(1, int(round(fps_in / fps_out)))

    max_frames = int(max(1.0, float(args.max_s)) * fps_out)

    frames = []
    idx = 0
    kept = 0
    while kept < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _resize_rgb(rgb, int(args.width) if int(args.width) > 0 else None)
            frames.append(rgb)
            kept += 1
        idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {in_path}")

    duration = 1.0 / fps_out
    imageio.mimsave(out_path, frames, format="GIF", duration=duration, loop=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

