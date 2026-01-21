from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import msgpackrpc
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.import_airsim import import_airsim
from bridge.overlay import OverlayLine, decode_png, draw_overlay_panel, encode_png
from bridge.paths import settings_json_path
from bridge.runpack import prepare_run, write_json
from bridge.safety import pick_best, score_pose_path
from bridge.video import build_split_video


def _write_settings(template_path: Path) -> Path:
    dst = settings_json_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _sim_get_images_with_retry(client, requests, *, vehicle_name: str, retries: int = 5, base_sleep_s: float = 0.2):
    for attempt in range(int(retries)):
        try:
            return client.simGetImages(requests, vehicle_name=vehicle_name)
        except msgpackrpc.error.TimeoutError:
            if attempt >= int(retries) - 1:
                return None
            time.sleep(float(base_sleep_s) * float(attempt + 1))
        except Exception:
            if attempt >= int(retries) - 1:
                return None
            time.sleep(float(base_sleep_s) * float(attempt + 1))


def _write_png_with_overlay(path: Path, png_bytes: bytes, overlay_lines: List[OverlayLine]) -> None:
    if not overlay_lines:
        path.write_bytes(png_bytes)
        return
    img = decode_png(png_bytes)
    if img is None:
        path.write_bytes(png_bytes)
        return
    draw_overlay_panel(img, overlay_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_png(path, img)


def _plot_user_point(client, airsim, *, x_ned: float, y_ned: float, z_ned_ground: float, label: str) -> None:
    p0 = airsim.Vector3r(float(x_ned), float(y_ned), float(z_ned_ground))
    p1 = airsim.Vector3r(float(x_ned), float(y_ned), float(z_ned_ground) - 30.0)
    try:
        client.simPlotLineList([p0, p1], color_rgba=[1.0, 0.2, 0.2, 1.0], thickness=10.0, duration=-1, is_persistent=True)
        client.simPlotPoints([p0], color_rgba=[1.0, 0.2, 0.2, 1.0], size=30.0, duration=-1, is_persistent=True)
        client.simPlotStrings([label], [p1], scale=10, color_rgba=[1.0, 1.0, 0.2, 1.0], duration=-1)
    except Exception:
        pass


def _required_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in timeseries: {missing}. Available={list(df.columns)}")


def _fmt_opt(row: pd.Series, key: str, fmt: str) -> str | None:
    if key not in row:
        return None
    try:
        return fmt.format(float(row[key]))
    except Exception:
        return None


def _opt_float(row: pd.Series, key: str) -> float | None:
    if key not in row:
        return None
    try:
        v = row[key]
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    except Exception:
        return None


def _metric_payload(row: pd.Series) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for key in [
        "S",
        "E_total",
        "E_mov",
        "E_tr",
        "E_comp",
        "D_total",
        "D_tr",
        "D_comp",
        "D_q",
        "D_uavq",
    ]:
        out[key] = _opt_float(row, key)
    # Keep constraint/violation bits if present (useful for debugging/overlay parity).
    for k in row.index:
        if isinstance(k, str) and k.startswith("viol_"):
            try:
                out[k] = float(int(row[k]))
            except Exception:
                out[k] = None
    return out


def main() -> int:
    logging.getLogger("tornado.general").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Replay a Framework timeseries into AirSim mainline (ExternalPhysicsEngine), with optional richer overlay (no changes to base scripts)."
    )
    parser.add_argument("--framework_timeseries", required=True)

    parser.add_argument("--output_root", default="runs_airsim")
    parser.add_argument("--settings_template", default="configs/airsim_settings/settings_mainline.json")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout_s", type=int, default=30)
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument("--ignore_collision", action="store_true")

    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)

    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--z_col", default="z")
    parser.add_argument("--yaw_col", default="yaw")
    parser.add_argument("--roll_col", default="roll")
    parser.add_argument("--pitch_col", default="pitch")

    parser.add_argument("--scale_xy", type=float, default=1.0)
    parser.add_argument("--scale_z", type=float, default=1.0)
    parser.add_argument("--x_offset_ned", type=float, default=0.0)
    parser.add_argument("--y_offset_ned", type=float, default=0.0)
    parser.add_argument("--z_offset_ned", type=float, default=0.0)
    parser.add_argument("--auto_offset", action="store_true")
    parser.add_argument("--x0", type=float, default=None)
    parser.add_argument("--y0", type=float, default=None)
    parser.add_argument("--z_up_m", type=float, default=None)
    parser.add_argument("--base_z_up_m", type=float, default=10.0)

    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--plot_user_point", action="store_true")
    parser.add_argument("--user_label", default="USER")
    parser.add_argument("--use_yaw", action="store_true")
    parser.add_argument("--use_rpy", action="store_true", help="If set and columns exist, set roll/pitch/yaw each step.")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    output_root = (base_dir / args.output_root).resolve()
    settings_template = (base_dir / args.settings_template).resolve()
    settings_path = _write_settings(settings_template)
    print(f"Wrote AirSim settings: {settings_path} (restart AirSim to apply changes)")

    df0 = pd.read_parquet(Path(args.framework_timeseries).expanduser().resolve())
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")
    df = df0.iloc[:: int(args.stride)].reset_index(drop=True)
    _required_columns(df, [args.x_col, args.y_col])
    has_z = args.z_col in df.columns
    has_yaw = args.yaw_col in df.columns
    has_rp = args.roll_col in df.columns and args.pitch_col in df.columns
    has_target_cols = all(c in df.columns for c in ["target_x", "target_y"])

    n_total = len(df)
    if n_total == 0:
        raise ValueError("Empty timeseries.")
    n = n_total
    if args.max_steps and int(args.max_steps) > 0:
        n = min(n, int(args.max_steps))
        df = df.iloc[:n].copy()

    x_first = float(df.loc[0, args.x_col])
    y_first = float(df.loc[0, args.y_col])
    z_first = float(df.loc[0, args.z_col]) if has_z else 0.0

    airsim = import_airsim()
    ctx = prepare_run(
        output_root=output_root,
        name="airsim_replay_mainline_plus",
        extra_meta={
            "mode": "replay_mainline_plus",
            "framework_timeseries": str(Path(args.framework_timeseries).expanduser().resolve()),
            "settings_template": str(settings_template),
            "settings_path": str(settings_path),
            "ip": args.ip,
            "port": args.port,
            "vehicle": args.vehicle,
            "stride": int(args.stride),
            "steps": int(n),
            "dt": float(args.dt),
            "fps": float(args.fps),
            "scale_xy": float(args.scale_xy),
            "scale_z": float(args.scale_z),
            "offset_ned": [float(args.x_offset_ned), float(args.y_offset_ned), float(args.z_offset_ned)],
            "auto_offset": bool(args.auto_offset),
            "anchor_xy": [args.x0, args.y0],
            "z_up_m": args.z_up_m,
            "base_z_up_m": float(args.base_z_up_m),
            "use_yaw": bool(args.use_yaw) and bool(has_yaw),
            "use_rpy": bool(args.use_rpy) and bool(has_rp) and bool(has_yaw),
            "ignore_collision": bool(args.ignore_collision),
            "overlay": bool(args.overlay),
            "plot_user_point": bool(args.plot_user_point),
        },
    )

    (ctx.run_dir / "airsim_settings.json").write_text(settings_template.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts_dir = ctx.run_dir / "artifacts"
    frames_fpv = artifacts_dir / "frames_fpv"
    frames_chase = artifacts_dir / "frames_chase"
    frames_fpv.mkdir(parents=True, exist_ok=True)
    frames_chase.mkdir(parents=True, exist_ok=True)

    client = airsim.VehicleClient(ip=args.ip, port=args.port, timeout_value=int(args.timeout_s))
    try:
        client.confirmConnection()
    except Exception as e:
        print("")
        print("ERROR: Failed to connect to AirSim RPC server for replay_mainline_plus.")
        print("- Requires AirSim restarted with ExternalPhysicsEngine enabled.")
        print(f"- settings path: {settings_path}")
        print(f"- template used: {settings_template}")
        print(f"- ip={args.ip} port={args.port}")
        print("")
        print(f"Exception: {type(e).__name__}: {e}")
        return 2

    ignore_collision = bool(args.ignore_collision)
    pose = client.simGetVehiclePose(vehicle_name=args.vehicle)
    if args.x0 is not None:
        pose.position.x_val = float(args.x0)
    if args.y0 is not None:
        pose.position.y_val = float(args.y0)
    base_z_ned = -float(args.z_up_m) if args.z_up_m is not None else -float(args.base_z_up_m)
    pose.position.z_val = float(base_z_ned)
    pose.orientation = airsim.to_quaternion(0.0, 0.0, 0.0)
    client.simSetVehiclePose(pose, ignore_collision, vehicle_name=args.vehicle)

    x_anchor = float(pose.position.x_val)
    y_anchor = float(pose.position.y_val)
    z_anchor = float(pose.position.z_val)

    if bool(args.auto_offset):
        import random

        rng = random.Random(str(ctx.run_id))
        offset_candidates = []
        for _i in range(10):
            x_off = float(args.x_offset_ned) + rng.uniform(-10.0, 20.0)
            y_off = float(args.y_offset_ned) + rng.choice([-1.0, 1.0]) * rng.uniform(18.0, 34.0)
            offset_candidates.append((x_off, y_off))

        sample_steps = min(int(n), 80)
        sample_idx = [int(round(i * (sample_steps - 1) / 9.0)) for i in range(10)] if sample_steps > 1 else [0]

        def scorer(off_xy):
            ox, oy = off_xy
            pose_pts = []
            for si in sample_idx:
                row = df.loc[si]
                x_fw = float(row[args.x_col])
                y_fw = float(row[args.y_col])
                z_fw = float(row[args.z_col]) if has_z else z_first
                x_ned = x_anchor + (x_fw - x_first) * float(args.scale_xy) + float(ox)
                y_ned = y_anchor + (y_fw - y_first) * float(args.scale_xy) + float(oy)
                if args.z_up_m is not None:
                    z_ned = -float(args.z_up_m) + float(args.z_offset_ned)
                else:
                    z_ned = z_anchor - (z_fw - z_first) * float(args.scale_z) + float(args.z_offset_ned)
                pose_pts.append((float(x_ned), float(y_ned), float(z_ned), 0.0, 0.0, 0.0, 0.0))
            return score_pose_path(client=client, airsim=airsim, vehicle_name=args.vehicle, pose_points=pose_pts, sample_every=1)

        (best_off, best_score) = pick_best(candidates=offset_candidates, scorer=scorer)
        args.x_offset_ned = float(best_off[0])
        args.y_offset_ned = float(best_off[1])
        write_json(
            ctx.run_dir / "scene_offset.json",
            {
                "x_offset_ned": float(args.x_offset_ned),
                "y_offset_ned": float(args.y_offset_ned),
                "score": {"collisions": int(best_score.collisions), "max_penetration_depth": float(best_score.max_penetration_depth)},
            },
        )

    # Choose one user point in Framework coords then map to NED for yaw-facing fallback.
    if has_target_cols:
        user_fw_x = float(df.loc[0, "target_x"])
        user_fw_y = float(df.loc[0, "target_y"])
        user_fw_z = float(df.loc[0, "target_z"]) if "target_z" in df.columns else z_first
    else:
        user_fw_x, user_fw_y, user_fw_z = x_first, y_first, z_first
    user_x_ned = x_anchor + (user_fw_x - x_first) * float(args.scale_xy) + float(args.x_offset_ned)
    user_y_ned = y_anchor + (user_fw_y - y_first) * float(args.scale_xy) + float(args.y_offset_ned)
    user_z_ned_ground = 0.0 + float(args.z_offset_ned)
    if bool(args.plot_user_point):
        _plot_user_point(client, airsim, x_ned=user_x_ned, y_ned=user_y_ned, z_ned_ground=user_z_ned_ground, label=str(args.user_label))

    rows: List[Dict[str, Any]] = []
    collision_count = 0
    capture_failures = 0
    t0 = time.time()
    prev_x_ned: float | None = None
    prev_y_ned: float | None = None

    for step in range(n):
        row = df.loc[step]
        x_fw = float(row[args.x_col])
        y_fw = float(row[args.y_col])
        z_fw = float(row[args.z_col]) if has_z else z_first

        x_ned = x_anchor + (x_fw - x_first) * float(args.scale_xy) + float(args.x_offset_ned)
        y_ned = y_anchor + (y_fw - y_first) * float(args.scale_xy) + float(args.y_offset_ned)
        if args.z_up_m is not None:
            z_ned = -float(args.z_up_m) + float(args.z_offset_ned)
        else:
            z_ned = z_anchor - (z_fw - z_first) * float(args.scale_z) + float(args.z_offset_ned)

        yaw_face_target = math.atan2(float(user_y_ned) - float(y_ned), float(user_x_ned) - float(x_ned))
        roll_cmd = 0.0
        pitch_cmd = 0.0
        yaw_cmd = float(yaw_face_target)
        if args.use_yaw and has_yaw:
            yaw_val = float(row[args.yaw_col])
            if math.isfinite(yaw_val):
                yaw_cmd = yaw_val
        if args.use_rpy and has_rp and has_yaw:
            roll_val = float(row[args.roll_col])
            pitch_val = float(row[args.pitch_col])
            yaw_val = float(row[args.yaw_col])
            if math.isfinite(roll_val):
                roll_cmd = roll_val
            if math.isfinite(pitch_val):
                pitch_cmd = pitch_val
            if math.isfinite(yaw_val):
                yaw_cmd = yaw_val

        pose.position.x_val = float(x_ned)
        pose.position.y_val = float(y_ned)
        pose.position.z_val = float(z_ned)
        pose.orientation = airsim.to_quaternion(float(pitch_cmd), float(roll_cmd), float(yaw_cmd))
        client.simSetVehiclePose(pose, ignore_collision, vehicle_name=args.vehicle)

        fpv_path = frames_fpv / f"frame_{step:06d}.png"
        chase_path = frames_chase / f"frame_{step:06d}.png"
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
            airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        ]
        responses = _sim_get_images_with_retry(client, requests, vehicle_name=args.vehicle)
        if responses is None:
            capture_failures += 1
            time.sleep(float(args.dt))
            continue

        fpv_bytes = responses[0].image_data_uint8
        chase_bytes = responses[1].image_data_uint8

        overlay_lines: List[OverlayLine] = []
        if bool(args.overlay):
            dx = float(x_ned) - float(user_x_ned)
            dy = float(y_ned) - float(user_y_ned)
            dist = float((dx * dx + dy * dy) ** 0.5)
            vx = 0.0
            vy = 0.0
            if prev_x_ned is not None and prev_y_ned is not None and float(args.dt) > 1e-9:
                vx = (float(x_ned) - float(prev_x_ned)) / float(args.dt)
                vy = (float(y_ned) - float(prev_y_ned)) / float(args.dt)
            speed = math.hypot(float(vx), float(vy))

            overlay_lines += [
                OverlayLine("MAINLINE REPLAY (pose-driven)", (255, 255, 255), scale=1.05),
                OverlayLine(f"step={step}/{n - 1}   t={float(step) * float(args.dt):.2f}s", (220, 220, 220)),
                OverlayLine(f"pos_ned=({float(x_ned):.1f},{float(y_ned):.1f},{float(z_ned):.1f})", (255, 255, 0)),
                OverlayLine(f"speed_xy≈{float(speed):.1f} m/s   dist_to_user≈{float(dist):.1f} m", (0, 255, 255)),
            ]

            # Optional metrics from Framework row, if present.
            for key, label in [
                ("S", "S"),
                ("E_total", "E_total"),
                ("D_total", "D_total"),
                ("E_mov", "E_mov"),
                ("E_tr", "E_tr"),
                ("E_comp", "E_comp"),
                ("D_tr", "D_tr"),
                ("D_comp", "D_comp"),
                ("D_q", "D_q"),
                ("D_uavq", "D_uavq"),
            ]:
                s = _fmt_opt(row, key, "{:.3g}")
                if s is not None:
                    overlay_lines.append(OverlayLine(f"{label}={s}", (200, 200, 200)))

            viol_keys = [k for k in df.columns if k.startswith("viol_")]
            if viol_keys:
                viol_bits = []
                for k in sorted(viol_keys):
                    try:
                        v = int(row[k])
                    except Exception:
                        continue
                    if v:
                        viol_bits.append(f"{k}={v}")
                if viol_bits:
                    overlay_lines.append(OverlayLine("VIOL: " + " ".join(viol_bits[:6]), (0, 0, 255)))

            overlay_lines.append(
                OverlayLine(
                    f"scale_xy={float(args.scale_xy):.3f} off=({float(args.x_offset_ned):+.1f},{float(args.y_offset_ned):+.1f},{float(args.z_offset_ned):+.1f})",
                    (200, 200, 200),
                )
            )

        _write_png_with_overlay(fpv_path, fpv_bytes, overlay_lines)
        _write_png_with_overlay(chase_path, chase_bytes, overlay_lines)

        collision = client.simGetCollisionInfo(vehicle_name=args.vehicle)
        if getattr(collision, "has_collided", False):
            collision_count += 1

        rows.append(
            {
                "run_id": ctx.run_id,
                "mode": "replay_mainline_plus",
                "step": step,
                "time_s": float(step) * float(args.dt),
                "x_ned": float(x_ned),
                "y_ned": float(y_ned),
                "z_ned": float(z_ned),
                "yaw_cmd": float(yaw_cmd),
                "roll_cmd": float(roll_cmd),
                "pitch_cmd": float(pitch_cmd),
                "has_collided": bool(getattr(collision, "has_collided", False)),
                **_metric_payload(row),
            }
        )
        prev_x_ned = float(x_ned)
        prev_y_ned = float(y_ned)
        time.sleep(float(args.dt))

    runtime_s = time.time() - t0

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(ctx.run_dir / "timeseries.parquet", index=False)

    video_path = artifacts_dir / "video.mp4"
    frames_written, width_px = build_split_video(
        frames_left_dir=frames_fpv,
        frames_right_dir=frames_chase,
        output_path=video_path,
        fps=args.fps,
    )
    summary = {
        "run_id": ctx.run_id,
        "mode": "replay_mainline_plus",
        "steps": int(n),
        "dt": float(args.dt),
        "fps": float(args.fps),
        "runtime_s": float(runtime_s),
        "capture_failures": int(capture_failures),
        "collision_count": int(collision_count),
        "frames_written": int(frames_written),
        "video_width_px": int(width_px),
    }
    write_json(ctx.run_dir / "summary.json", summary)
    print(f"Output: {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
