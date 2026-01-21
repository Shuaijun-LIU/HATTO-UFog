from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import msgpackrpc
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _capture_dual_rgb(airsim, client, *, vehicle_name: str, fpv_path: Path, chase_path: Path) -> List[bytes]:
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
    ]
    responses = client.simGetImages(requests, vehicle_name=vehicle_name)
    airsim.write_file(str(fpv_path), responses[0].image_data_uint8)
    airsim.write_file(str(chase_path), responses[1].image_data_uint8)
    return [responses[0].image_data_uint8, responses[1].image_data_uint8]


def _sim_get_images_with_retry(
    client,
    requests,
    *,
    vehicle_name: str,
    retries: int = 5,
    base_sleep_s: float = 0.2,
):
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


def _required_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in timeseries: {missing}. Available={list(df.columns)}")


def main() -> int:
    logging.getLogger("tornado.general").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Replay a Framework `timeseries.parquet` into AirSim mainline (ExternalPhysicsEngine) and record dual-view video."
    )
    parser.add_argument("--framework_timeseries", required=True, help="Path to Framework timeseries.parquet")

    parser.add_argument("--output_root", default="runs_airsim")
    parser.add_argument("--settings_template", default="configs/airsim_settings/settings_mainline.json")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout_s", type=int, default=30)
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument("--ignore_collision", action="store_true")
    parser.add_argument("--overlay", action="store_true", help="Overlay real-time status text on captured frames.")
    parser.add_argument("--plot_user_point", action="store_true", help="Plot a visible user/target marker in the scene.")
    parser.add_argument("--user_label", default="USER", help="Label for the target/user point marker.")
    parser.add_argument("--user_fw_x", type=float, default=None, help="Optional: user point X in Framework coordinates.")
    parser.add_argument("--user_fw_y", type=float, default=None, help="Optional: user point Y in Framework coordinates.")
    parser.add_argument("--user_fw_z", type=float, default=None, help="Optional: user point Z in Framework coordinates.")
    parser.add_argument(
        "--time_of_day",
        default="",
        help="Optional: set time-of-day via simSetTimeOfDay. Example: '2020-01-01 18:00:00'.",
    )
    parser.add_argument("--weather_fog", type=float, default=None, help="Optional: enable weather and set Fog [0..1].")
    parser.add_argument("--weather_rain", type=float, default=None, help="Optional: enable weather and set Rain [0..1].")
    parser.add_argument("--weather_snow", type=float, default=None, help="Optional: enable weather and set Snow [0..1].")
    parser.add_argument("--weather_road_snow", type=float, default=None, help="Optional: enable weather and set RoadSnow [0..1].")

    parser.add_argument("--x_col", default="x", help="Framework x column (meters)")
    parser.add_argument("--y_col", default="y", help="Framework y column (meters)")
    parser.add_argument("--z_col", default="z", help="Framework z-up column (meters)")
    parser.add_argument("--yaw_col", default="yaw", help="Optional: yaw column (radians)")

    parser.add_argument("--stride", type=int, default=1, help="Downsample: take every N rows from the Framework timeseries.")
    parser.add_argument("--max_steps", type=int, default=0, help="0 means all steps after stride.")
    parser.add_argument("--dt", type=float, default=0.05, help="Wall-clock seconds per step (and capture interval).")
    parser.add_argument("--fps", type=float, default=20.0)

    parser.add_argument("--scale_xy", type=float, default=1.0, help="Scale factor applied to (x,y) displacements.")
    parser.add_argument("--scale_z", type=float, default=1.0, help="Scale factor applied to z displacement.")
    parser.add_argument("--x_offset_ned", type=float, default=0.0, help="Additive offset after mapping, NED meters.")
    parser.add_argument("--y_offset_ned", type=float, default=0.0)
    parser.add_argument("--z_offset_ned", type=float, default=0.0)
    parser.add_argument(
        "--auto_offset",
        action="store_true",
        help="Auto-pick an off-road scene offset (demo variety without manual point-picking).",
    )

    parser.add_argument("--x0", type=float, default=None, help="Optional: anchor X (NED meters). Defaults to current spawn X.")
    parser.add_argument("--y0", type=float, default=None, help="Optional: anchor Y (NED meters). Defaults to current spawn Y.")
    parser.add_argument(
        "--z_up_m",
        type=float,
        default=None,
        help="Optional: force constant altitude (Framework z ignored). If omitted, uses Framework z (relative to first point).",
    )
    parser.add_argument("--base_z_up_m", type=float, default=10.0, help="Anchor altitude when z_up_m is not provided.")

    parser.add_argument("--use_yaw", action="store_true", help="If set and yaw_col exists, set yaw each step (radians).")
    parser.add_argument(
        "--screenshot_stride",
        type=int,
        default=30,
        help="Write chase screenshots every N steps into artifacts/screenshots (0 disables).",
    )
    args = parser.parse_args()

    framework_timeseries = Path(args.framework_timeseries).expanduser().resolve()
    if not framework_timeseries.exists():
        raise FileNotFoundError(framework_timeseries)

    base_dir = Path(__file__).resolve().parents[1]
    output_root = (base_dir / args.output_root).resolve()
    settings_template = (base_dir / args.settings_template).resolve()
    settings_path = _write_settings(settings_template)
    print(f"Wrote AirSim settings: {settings_path} (restart AirSim to apply changes)")

    df0 = pd.read_parquet(framework_timeseries)
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")
    df = df0.iloc[:: int(args.stride)].reset_index(drop=True)
    _required_columns(df, [args.x_col, args.y_col])
    has_z = args.z_col in df.columns
    has_yaw = args.yaw_col in df.columns
    has_target_cols = all(c in df.columns for c in ["target_x", "target_y"])

    n_total = len(df)
    if n_total == 0:
        raise ValueError(f"Empty timeseries: {framework_timeseries}")
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
        name="airsim_replay_mainline",
        extra_meta={
            "mode": "replay_mainline",
            "framework_timeseries": str(framework_timeseries),
            "settings_template": str(settings_template),
            "settings_path": str(settings_path),
            "ip": args.ip,
            "port": args.port,
            "vehicle": args.vehicle,
            "stride": int(args.stride),
            "max_steps": int(args.max_steps),
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
            "ignore_collision": bool(args.ignore_collision),
            "screenshot_stride": int(args.screenshot_stride),
            "time_of_day": args.time_of_day,
            "weather_fog": args.weather_fog,
            "weather_rain": args.weather_rain,
            "weather_snow": args.weather_snow,
            "weather_road_snow": args.weather_road_snow,
            "overlay": bool(args.overlay),
            "plot_user_point": bool(args.plot_user_point),
            "user_label": str(args.user_label),
        },
    )

    (ctx.run_dir / "airsim_settings.json").write_text(settings_template.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts_dir = ctx.run_dir / "artifacts"
    frames_fpv = artifacts_dir / "frames_fpv"
    frames_chase = artifacts_dir / "frames_chase"
    frames_fpv.mkdir(parents=True, exist_ok=True)
    frames_chase.mkdir(parents=True, exist_ok=True)
    screenshots_dir = artifacts_dir / "screenshots"
    if int(args.screenshot_stride) > 0:
        screenshots_dir.mkdir(parents=True, exist_ok=True)

    client = airsim.VehicleClient(ip=args.ip, port=args.port, timeout_value=int(args.timeout_s))
    try:
        client.confirmConnection()
    except Exception as e:
        print("")
        print("ERROR: Failed to connect to AirSim RPC server for replay_mainline.")
        print("- Replay mainline requires AirSim to be restarted with ExternalPhysicsEngine enabled.")
        print(f"- settings path: {settings_path}")
        print(f"- template used: {settings_template}")
        print(f"- ip={args.ip} port={args.port}")
        print("")
        print(f"Exception: {type(e).__name__}: {e}")
        return 2

    if args.time_of_day:
        try:
            client.simSetTimeOfDay(True, start_datetime=str(args.time_of_day), is_start_datetime_dst=False, celestial_clock_speed=1)
        except Exception:
            pass
    if args.weather_fog is not None or args.weather_rain is not None or args.weather_snow is not None or args.weather_road_snow is not None:
        try:
            client.simEnableWeather(True)
            if args.weather_fog is not None:
                client.simSetWeatherParameter(airsim.WeatherParameter.Fog, float(args.weather_fog))
            if args.weather_rain is not None:
                client.simSetWeatherParameter(airsim.WeatherParameter.Rain, float(args.weather_rain))
            if args.weather_snow is not None:
                client.simSetWeatherParameter(airsim.WeatherParameter.Snow, float(args.weather_snow))
            if args.weather_road_snow is not None:
                client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, float(args.weather_road_snow))
        except Exception:
            pass

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
        # Generate a handful of offset candidates; we'll pick the one with least collision penetration
        # by teleport-sampling a few mapped points.
        offset_candidates: List[Tuple[float, float]] = []
        for i in range(10):
            x_off = float(args.x_offset_ned) + rng.uniform(-10.0, 20.0)
            y_off = float(args.y_offset_ned) + rng.choice([-1.0, 1.0]) * rng.uniform(18.0, 34.0)
            offset_candidates.append((x_off, y_off))

        sample_steps = min(int(n), 80)
        sample_idx = [int(round(i * (sample_steps - 1) / 9.0)) for i in range(10)] if sample_steps > 1 else [0]

        def scorer(off_xy: Tuple[float, float]):
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

    # Choose one "user point" (target marker) in Framework coordinates, then map to NED.
    if args.user_fw_x is not None and args.user_fw_y is not None:
        user_fw_x = float(args.user_fw_x)
        user_fw_y = float(args.user_fw_y)
        user_fw_z = float(args.user_fw_z) if args.user_fw_z is not None else z_first
    elif has_target_cols:
        user_fw_x = float(df.loc[0, "target_x"])
        user_fw_y = float(df.loc[0, "target_y"])
        user_fw_z = float(df.loc[0, "target_z"]) if "target_z" in df.columns else z_first
    else:
        user_fw_x = x_first
        user_fw_y = y_first
        user_fw_z = z_first

    user_x_ned = x_anchor + (user_fw_x - x_first) * float(args.scale_xy) + float(args.x_offset_ned)
    user_y_ned = y_anchor + (user_fw_y - y_first) * float(args.scale_xy) + float(args.y_offset_ned)
    user_z_ned_ground = 0.0 + float(args.z_offset_ned)
    if bool(args.plot_user_point):
        _plot_user_point(
            client,
            airsim,
            x_ned=float(user_x_ned),
            y_ned=float(user_y_ned),
            z_ned_ground=float(user_z_ned_ground),
            label=str(args.user_label),
        )

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

        pose.position.x_val = float(x_ned)
        pose.position.y_val = float(y_ned)
        pose.position.z_val = float(z_ned)
        # Make the motion look purposeful: if we don't have a yaw command, face the target marker.
        yaw_face_target = math.atan2(float(user_y_ned) - float(y_ned), float(user_x_ned) - float(x_ned))
        if args.use_yaw and has_yaw:
            yaw_cmd = float(row[args.yaw_col])
        else:
            yaw_cmd = float(yaw_face_target)
        pose.orientation = airsim.to_quaternion(0.0, 0.0, float(yaw_cmd))
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
            collided = bool(getattr(client.simGetCollisionInfo(vehicle_name=args.vehicle), "has_collided", False))
            overlay_lines = [
                OverlayLine("REPLAY MAINLINE (Framework→AirSim)", (255, 255, 255), scale=1.05),
                OverlayLine(f"step={step}/{n - 1}   t={float(step) * float(args.dt):.2f}s", (220, 220, 220)),
                OverlayLine(
                    f"pos_ned=({float(x_ned):.1f},{float(y_ned):.1f},{float(z_ned):.1f})   yaw={float(yaw_cmd):+.2f} rad",
                    (255, 255, 0),
                ),
                OverlayLine(f"vel_xy=({float(vx):+.1f},{float(vy):+.1f}) m/s   speed={float(speed):.1f}", (0, 255, 0)),
                OverlayLine(f"user_ned=({float(user_x_ned):.1f},{float(user_y_ned):.1f},{float(user_z_ned_ground):.1f})   dist={dist:.1f} m", (0, 255, 255)),
                OverlayLine(f"scale_xy={float(args.scale_xy):.3f}  offset_ned=({float(args.x_offset_ned):+.1f},{float(args.y_offset_ned):+.1f},{float(args.z_offset_ned):+.1f})", (200, 200, 200)),
                OverlayLine(
                    "ENV: "
                    f"time={args.time_of_day or 'default'}  "
                    f"fog={args.weather_fog if args.weather_fog is not None else 0.0:.2f}  "
                    f"rain={args.weather_rain if args.weather_rain is not None else 0.0:.2f}  "
                    f"snow={args.weather_snow if args.weather_snow is not None else 0.0:.2f}  "
                    f"roadSnow={args.weather_road_snow if args.weather_road_snow is not None else 0.0:.2f}",
                    (200, 200, 200),
                ),
                OverlayLine(f"collision={'YES' if collided else 'no'}", (0, 0, 255) if collided else (200, 200, 200)),
            ]

        _write_png_with_overlay(fpv_path, fpv_bytes, overlay_lines)
        _write_png_with_overlay(chase_path, chase_bytes, overlay_lines)

        if int(args.screenshot_stride) > 0 and (step % int(args.screenshot_stride) == 0):
            _write_png_with_overlay(screenshots_dir / f"chase_{step:06d}.png", chase_bytes, overlay_lines)

        collision = client.simGetCollisionInfo(vehicle_name=args.vehicle)
        if getattr(collision, "has_collided", False):
            collision_count += 1

        rows.append(
            {
                "run_id": ctx.run_id,
                "mode": "replay_mainline",
                "step": int(step),
                "time_s": float(step) * float(args.dt),
                "x_ned": float(x_ned),
                "y_ned": float(y_ned),
                "z_ned": float(z_ned),
                "user_x_ned": float(user_x_ned),
                "user_y_ned": float(user_y_ned),
                "user_z_ned_ground": float(user_z_ned_ground),
                "framework_x": float(x_fw),
                "framework_y": float(y_fw),
                "framework_z": float(z_fw) if has_z else None,
                "has_collided": bool(getattr(collision, "has_collided", False)),
            }
        )
        time.sleep(float(args.dt))
        prev_x_ned = float(x_ned)
        prev_y_ned = float(y_ned)

    runtime_s = time.time() - t0

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(ctx.run_dir / "timeseries.parquet", index=False)

    video_path = artifacts_dir / "video.mp4"
    frames_written, width_px = build_split_video(
        frames_left_dir=frames_fpv,
        frames_right_dir=frames_chase,
        output_path=video_path,
        fps=args.fps,
    )

    summary: Dict[str, Any] = {
        "run_id": ctx.run_id,
        "mode": "replay_mainline",
        "steps": int(n),
        "dt": float(args.dt),
        "fps": float(args.fps),
        "runtime_s": float(runtime_s),
        "collision_count": int(collision_count),
        "capture_failures": int(capture_failures),
        "framework_timeseries": str(framework_timeseries),
        "offset_ned": [float(args.x_offset_ned), float(args.y_offset_ned), float(args.z_offset_ned)],
        "video": {"path": str(video_path), "frames": int(frames_written), "width_px": int(width_px)},
    }
    write_json(ctx.run_dir / "summary.json", summary)
    (ctx.run_dir / "summary.txt").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Done:", ctx.run_dir)
    print("Video:", video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
