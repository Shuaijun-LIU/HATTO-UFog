from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sorted_pngs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png"))


def _sorted_frames(folder: Path) -> List[Path]:
    pngs = _sorted_pngs(folder)
    if pngs:
        return pngs
    return sorted(folder.glob("*.npy"))


def _read_frame_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".png":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read: {path}")
        return img  # BGR
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise RuntimeError(f"Unexpected npy frame shape: {arr.shape} for {path}")
        rgb = arr[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb[:, :, ::-1]  # BGR
    raise RuntimeError(f"Unsupported frame type: {path}")


def _ensure_same_height(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if left.shape[0] == right.shape[0]:
        return left, right
    height_px = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * height_px / left.shape[0]), height_px), interpolation=cv2.INTER_AREA)
    right = cv2.resize(
        right, (int(right.shape[1] * height_px / right.shape[0]), height_px), interpolation=cv2.INTER_AREA
    )
    return left, right


def _build_video_from_pngs(*, frames_dir: Path, output_path: Path, fps: float) -> int:
    paths = _sorted_pngs(frames_dir)
    if not paths:
        raise RuntimeError(f"No PNG frames found in: {frames_dir}")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {paths[0]}")
    height_px, width_px = int(first.shape[0]), int(first.shape[1])

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
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read: {p}")
            if img.shape[0] != height_px or img.shape[1] != width_px:
                img = cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    return len(paths)


def _build_video_from_frames(*, frames_dir: Path, output_path: Path, fps: float) -> int:
    paths = _sorted_frames(frames_dir)
    if not paths:
        raise RuntimeError(f"No frames found in: {frames_dir} (need .png or .npy)")

    first = _read_frame_any(paths[0])
    height_px, width_px = int(first.shape[0]), int(first.shape[1])

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
        for p in paths:
            img = _read_frame_any(p)
            if img.shape[0] != height_px or img.shape[1] != width_px:
                img = cv2.resize(img, (width_px, height_px), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    return len(paths)


def _build_split_video_any(
    *,
    frames_left_dir: Path,
    frames_right_dir: Path,
    output_path: Path,
    fps: float,
) -> Tuple[int, int]:
    left_paths = _sorted_frames(frames_left_dir)
    right_paths = _sorted_frames(frames_right_dir)
    if not left_paths or not right_paths:
        raise RuntimeError("No frames found (need both left/right .png or .npy sequences).")
    num_frames = min(len(left_paths), len(right_paths))

    first_left = _read_frame_any(left_paths[0])
    first_right = _read_frame_any(right_paths[0])
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
            left = _read_frame_any(left_paths[i])
            right = _read_frame_any(right_paths[i])
            left, right = _ensure_same_height(left, right)
            frame = cv2.hconcat([left, right])
            writer.write(frame)
    finally:
        writer.release()
    return num_frames, width_px


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _draw_trajectory_frame(
    *,
    canvas_wh: Tuple[int, int],
    xys: np.ndarray,
    zs: np.ndarray,
    t_s: np.ndarray,
    idx: int,
    margin: float = 0.15,
) -> np.ndarray:
    width_px, height_px = int(canvas_wh[0]), int(canvas_wh[1])
    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    xy = xys[: idx + 1]
    x_min, y_min = np.min(xys[:, 0]), np.min(xys[:, 1])
    x_max, y_max = np.max(xys[:, 0]), np.max(xys[:, 1])
    dx = float(max(1e-6, x_max - x_min))
    dy = float(max(1e-6, y_max - y_min))
    x_min -= dx * margin
    x_max += dx * margin
    y_min -= dy * margin
    y_max += dy * margin

    def to_px(x: float, y: float) -> Tuple[int, int]:
        u = int(round((x - x_min) / max(1e-9, x_max - x_min) * (width_px - 1)))
        v = int(round((1.0 - (y - y_min) / max(1e-9, y_max - y_min)) * (height_px - 1)))
        return u, v

    pts = np.array([to_px(float(p[0]), float(p[1])) for p in xy], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], isClosed=False, color=(30, 30, 30), thickness=2, lineType=cv2.LINE_AA)
    if len(pts) >= 1:
        cv2.circle(img, tuple(pts[-1, 0].tolist()), radius=6, color=(0, 102, 255), thickness=-1, lineType=cv2.LINE_AA)

    t_cur = float(t_s[idx]) if idx < len(t_s) else float(idx)
    z_cur = float(zs[idx]) if idx < len(zs) else 0.0
    cv2.putText(
        img,
        f"t={t_cur:.2f}s  z={z_cur:.2f}m",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Isaac Sim (fallback trajectory video)",
        (14, height_px - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 80, 80),
        2,
        cv2.LINE_AA,
    )
    return img


def _build_fallback_trajectory_video(
    *,
    timeseries_jsonl: Path,
    output_path: Path,
    screenshot_path: Path,
    fps: float,
    stride: int,
    canvas_wh: Tuple[int, int],
) -> Dict[str, Any]:
    rows = _read_jsonl(timeseries_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in: {timeseries_jsonl}")

    xs = np.array([_safe_float(r.get("x")) for r in rows], dtype=float)
    ys = np.array([_safe_float(r.get("y")) for r in rows], dtype=float)
    zs = np.array([_safe_float(r.get("z")) for r in rows], dtype=float)
    ts = np.array([_safe_float(r.get("t_s", r.get("time_s", r.get("t", 0.0)))) for r in rows], dtype=float)
    xys = np.stack([xs, ys], axis=1)

    width_px, height_px = int(canvas_wh[0]), int(canvas_wh[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width_px, height_px),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    frame_indices = list(range(0, len(rows), max(1, int(stride))))
    if frame_indices[-1] != len(rows) - 1:
        frame_indices.append(len(rows) - 1)

    last_frame: Optional[np.ndarray] = None
    try:
        for idx in frame_indices:
            frame = _draw_trajectory_frame(canvas_wh=canvas_wh, xys=xys, zs=zs, t_s=ts, idx=idx)
            writer.write(frame)
            last_frame = frame
    finally:
        writer.release()

    if last_frame is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(screenshot_path), last_frame)

    return {
        "mode": "trajectory_fallback",
        "frames": int(len(frame_indices)),
        "fps": float(fps),
        "stride": int(stride),
        "canvas_wh": [int(width_px), int(height_px)],
        "output_path": str(output_path),
        "screenshot_path": str(screenshot_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isaac Sim postprocess: build MP4 from captured frames (preferred) or fallback trajectory video."
    )
    parser.add_argument("--run_dir", required=True, help="runs_isaac/<run_id>__<slug> directory.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=2, help="Stride for fallback trajectory video (>=1).")
    parser.add_argument("--canvas", default="1280x720", help="Fallback canvas size, e.g. 1280x720.")
    parser.add_argument("--force_fallback", action="store_true", help="Ignore captured frames and build fallback video.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    artifacts_dir = run_dir / "artifacts"
    frames_fpv = artifacts_dir / "frames_fpv"
    frames_chase = artifacts_dir / "frames_chase"
    single_dir = artifacts_dir / "frames"
    video_path = artifacts_dir / "video.mp4"
    screenshot_path = artifacts_dir / "screenshot.png"
    screenshot_raw = artifacts_dir / "screenshot_raw.png"

    try:
        width_str, height_str = str(args.canvas).lower().split("x", 1)
        canvas_wh = (int(width_str), int(height_str))
    except Exception:
        raise SystemExit(f"Invalid --canvas: {args.canvas} (expected like 1280x720)")

    meta: Dict[str, Any] = {"run_dir": str(run_dir)}

    if not args.force_fallback:
        if frames_fpv.exists() and frames_chase.exists() and _sorted_frames(frames_fpv) and _sorted_frames(frames_chase):
            frames_written, width_px = _build_split_video_any(
                frames_left_dir=frames_fpv,
                frames_right_dir=frames_chase,
                output_path=video_path,
                fps=float(args.fps),
            )
            meta.update(
                {
                    "mode": "split_frames",
                    "frames": int(frames_written),
                    "fps": float(args.fps),
                    "width_px": int(width_px),
                    "video_path": str(video_path),
                }
            )
            # Use the first left frame as screenshot.
            first = _sorted_frames(frames_fpv)[0]
            img = _read_frame_any(first)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(screenshot_path), img)
            meta["screenshot_path"] = str(screenshot_path)
        elif single_dir.exists() and _sorted_frames(single_dir):
            frames = _build_video_from_frames(frames_dir=single_dir, output_path=video_path, fps=float(args.fps))
            meta.update(
                {"mode": "single_frames", "frames": int(frames), "fps": float(args.fps), "video_path": str(video_path)}
            )
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            if screenshot_raw.exists():
                img = _read_frame_any(screenshot_raw)
                cv2.imwrite(str(screenshot_path), img)
                meta["screenshot_path"] = str(screenshot_path)
                meta["screenshot_source"] = str(screenshot_raw)
            else:
                first = _sorted_frames(single_dir)[0]
                img = _read_frame_any(first)
                cv2.imwrite(str(screenshot_path), img)
                meta["screenshot_path"] = str(screenshot_path)
                meta["screenshot_source"] = str(first)

    if "mode" not in meta:
        timeseries_jsonl = run_dir / "timeseries.jsonl"
        if not timeseries_jsonl.exists():
            raise SystemExit(
                "No captured frames found and no timeseries.jsonl for fallback.\n"
                f"- looked for: {frames_fpv}, {frames_chase}, {single_dir}\n"
                f"- and: {timeseries_jsonl}"
            )
        meta.update(
            _build_fallback_trajectory_video(
                timeseries_jsonl=timeseries_jsonl,
                output_path=video_path,
                screenshot_path=screenshot_path,
                fps=float(args.fps),
                stride=int(max(1, args.stride)),
                canvas_wh=canvas_wh,
            )
        )

    (artifacts_dir / "video_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[ok] video:", video_path)
    print("[ok] screenshot:", screenshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
