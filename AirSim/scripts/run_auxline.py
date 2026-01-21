from __future__ import annotations

import argparse
import json
import logging
import math
import msgpackrpc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.flightplan import choose_flightplan, sample_showcase_path_xy
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


def _capture_dual_rgb(airsim, client, *, vehicle_name: str, fpv_path: Path, chase_path: Path) -> None:
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
    ]
    responses = client.simGetImages(requests, vehicle_name=vehicle_name)
    airsim.write_file(str(fpv_path), responses[0].image_data_uint8)
    airsim.write_file(str(chase_path), responses[1].image_data_uint8)


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
        # Fallback: write raw bytes (may be empty/corrupted; video builder will skip unreadable frames).
        path.write_bytes(png_bytes)
        return
    draw_overlay_panel(img, overlay_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_png(path, img)


def _plot_user_point(client, airsim, *, x_ned: float, y_ned: float, z_ned_ground: float, label: str) -> None:
    # Visual marker that shows up in camera frames: a vertical line + point + text.
    p0 = airsim.Vector3r(float(x_ned), float(y_ned), float(z_ned_ground))
    p1 = airsim.Vector3r(float(x_ned), float(y_ned), float(z_ned_ground) - 30.0)  # 30m up (NED negative is up)
    try:
        client.simPlotLineList([p0, p1], color_rgba=[1.0, 0.2, 0.2, 1.0], thickness=10.0, duration=-1, is_persistent=True)
        client.simPlotPoints([p0], color_rgba=[1.0, 0.2, 0.2, 1.0], size=30.0, duration=-1, is_persistent=True)
        client.simPlotStrings([label], [p1], scale=10, color_rgba=[1.0, 1.0, 0.2, 1.0], duration=-1)
    except Exception:
        # Plot APIs may not be enabled on some builds; don't fail the run.
        pass


def main() -> int:
    logging.getLogger("tornado.general").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="AirSim auxline (AirSim dynamics) video run.")
    parser.add_argument("--output_root", default="runs_airsim")
    parser.add_argument("--settings_template", default="configs/airsim_settings/settings_auxline.json")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout_s", type=int, default=30)
    parser.add_argument("--vehicle", default="Drone1")

    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.05, help="Command duration per step (and capture interval).")
    parser.add_argument("--fps", type=float, default=20.0, help="Video FPS (may differ from 1/dt).")

    parser.add_argument("--speed_m_s", type=float, default=2.0)
    parser.add_argument("--wind_x", type=float, default=0.0, help="Global wind in NED, m/s (AirSim wind is global).")
    parser.add_argument("--wind_y", type=float, default=0.0)
    parser.add_argument("--wind_z", type=float, default=0.0)
    parser.add_argument("--user_label", default="USER", help="Label for the target/user point marker.")
    parser.add_argument("--plot_user_point", action="store_true", help="Plot a visible target/user point marker in the scene.")
    parser.add_argument("--overlay", action="store_true", help="Overlay real-time status text on captured frames.")
    parser.add_argument(
        "--scene_profile",
        default="airsimnh",
        help="Heuristic profile for auto-picking safer start/target (airsimnh|abandonedpark).",
    )
    parser.add_argument(
        "--time_of_day",
        default="",
        help="Optional: set time-of-day via simSetTimeOfDay. Example: '2020-01-01 18:00:00'.",
    )
    parser.add_argument("--weather_fog", type=float, default=None, help="Optional: enable weather and set Fog [0..1].")
    parser.add_argument("--weather_rain", type=float, default=None, help="Optional: enable weather and set Rain [0..1].")
    parser.add_argument("--weather_snow", type=float, default=None, help="Optional: enable weather and set Snow [0..1].")
    parser.add_argument("--weather_road_snow", type=float, default=None, help="Optional: enable weather and set RoadSnow [0..1].")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    output_root = (base_dir / args.output_root).resolve()
    settings_template = (base_dir / args.settings_template).resolve()
    settings_path = _write_settings(settings_template)
    print(f"Wrote AirSim settings: {settings_path} (restart AirSim to apply changes)")

    airsim = import_airsim()
    ctx = prepare_run(
        output_root=output_root,
        name="airsim_auxline",
        extra_meta={
            "mode": "auxline",
            "settings_template": str(settings_template),
            "settings_path": str(settings_path),
            "ip": args.ip,
            "port": args.port,
            "vehicle": args.vehicle,
            "steps": args.steps,
            "dt": args.dt,
            "fps": args.fps,
            "speed_m_s": args.speed_m_s,
            "wind_ned": [args.wind_x, args.wind_y, args.wind_z],
            "plot_user_point": bool(args.plot_user_point),
            "overlay": bool(args.overlay),
            "time_of_day": args.time_of_day,
            "weather_fog": args.weather_fog,
            "weather_rain": args.weather_rain,
            "weather_snow": args.weather_snow,
            "weather_road_snow": args.weather_road_snow,
        },
    )

    # Snapshot the exact settings used.
    (ctx.run_dir / "airsim_settings.json").write_text(settings_template.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts_dir = ctx.run_dir / "artifacts"
    frames_fpv = artifacts_dir / "frames_fpv"
    frames_chase = artifacts_dir / "frames_chase"
    frames_fpv.mkdir(parents=True, exist_ok=True)
    frames_chase.mkdir(parents=True, exist_ok=True)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port, timeout_value=int(args.timeout_s))
    try:
        client.confirmConnection()
    except Exception as e:
        print("")
        print("ERROR: Failed to connect to AirSim RPC server for auxline.")
        print("- Auxline expects AirSim running with normal physics (NOT ExternalPhysicsEngine).")
        print(f"- settings path: {settings_path}")
        print(f"- template used: {settings_template}")
        print(f"- ip={args.ip} port={args.port}")
        print("")
        print(f"Exception: {type(e).__name__}: {e}")
        return 2
    client.enableApiControl(True, vehicle_name=args.vehicle)
    client.armDisarm(True, vehicle_name=args.vehicle)

    # Takeoff (best-effort).
    try:
        client.takeoffAsync(vehicle_name=args.vehicle).join()
    except Exception:
        pass

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

    client.simSetWind(airsim.Vector3r(float(args.wind_x), float(args.wind_y), float(args.wind_z)))

    state0 = client.getMultirotorState(vehicle_name=args.vehicle)
    pos0 = state0.kinematics_estimated.position
    spawn_x = float(pos0.x_val)
    spawn_y = float(pos0.y_val)

    # Pick safer candidate plan by teleport-sampling collision penetration along a candidate path.
    candidates = [
        choose_flightplan(
            run_id=ctx.run_id,
            spawn_xy_ned=(spawn_x, spawn_y),
            prefer_offroad=True,
            scene_profile=str(args.scene_profile),
            seed_salt=str(i),
        )
        for i in range(12)
    ]

    def scorer(cand):
        z_ned_local = -float(cand.z_up_m)
        _target_xy, path = sample_showcase_path_xy(
            run_id=ctx.run_id,
            plan=cand,
            steps=max(60, int(args.steps)),
            speed_m_s=float(args.speed_m_s),
            dt_s=float(args.dt),
            face_target=True,
        )
        pose_pts = [(x, y, z_ned_local, 0.0, 0.0, yaw, 0.0) for (x, y, yaw) in path]
        return score_pose_path(client=client, airsim=airsim, vehicle_name=args.vehicle, pose_points=pose_pts, sample_every=14)

    fp, fp_score = pick_best(candidates=candidates, scorer=scorer)
    write_json(ctx.run_dir / "safety_score.json", {"collisions": fp_score.collisions, "max_penetration_depth": fp_score.max_penetration_depth})

    # For auxline we fly the path at a fixed altitude derived from the chosen plan.
    z_ned = -float(fp.z_up_m)
    target_xy, path_xy_yaw = sample_showcase_path_xy(
        run_id=ctx.run_id,
        plan=fp,
        steps=int(args.steps),
        speed_m_s=float(args.speed_m_s),
        dt_s=float(args.dt),
        face_target=True,
    )
    x0, y0 = float(fp.start_xy_ned[0]), float(fp.start_xy_ned[1])
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    if bool(args.plot_user_point):
        _plot_user_point(client, airsim, x_ned=target_x, y_ned=target_y, z_ned_ground=0.0, label=str(args.user_label))

    write_json(
        ctx.run_dir / "flightplan.json",
        {
            "spawn_xy_ned": [float(spawn_x), float(spawn_y)],
            "start_xy_ned": [float(x0), float(y0)],
            "target_xy_ned": [float(target_x), float(target_y)],
            "z_ned": float(z_ned),
            "bend_m": float(fp.bend_m),
            "orbit_radius_m": float(fp.orbit_radius_m),
            "orbit_turns": float(fp.orbit_turns),
            "curve_sign": float(fp.curve_sign),
        },
    )

    # Generate a smooth non-straight path, then fly it via moveOnPathAsync.
    # Sample a handful of waypoints (AirSim will smooth between them).
    if "path_xy_yaw" not in locals():
        _, path_xy_yaw = sample_showcase_path_xy(
            run_id=ctx.run_id,
            plan=fp,
            steps=int(args.steps),
            speed_m_s=float(args.speed_m_s),
            dt_s=float(args.dt),
            face_target=True,
        )
    waypoints = []
    n_wp = 10
    for j in range(n_wp):
        s_idx = int(round(float(j) / float(n_wp - 1) * float(len(path_xy_yaw) - 1)))
        x, y, _yaw = path_xy_yaw[s_idx]
        waypoints.append(airsim.Vector3r(float(x), float(y), float(z_ned)))

    # Teleport to a slightly varied start to avoid identical demos.
    try:
        pose0 = client.simGetVehiclePose(vehicle_name=args.vehicle)
        pose0.position.x_val = float(x0)
        pose0.position.y_val = float(y0)
        pose0.position.z_val = float(z_ned)
        client.simSetVehiclePose(pose0, ignore_collision=True, vehicle_name=args.vehicle)
    except Exception:
        pass

    # Ensure we are at the planned altitude before starting path tracking.
    try:
        client.moveToZAsync(float(z_ned), velocity=max(2.0, float(args.speed_m_s)), vehicle_name=args.vehicle).join()
    except Exception:
        pass

    total_time = float(args.steps) * float(args.dt)
    path_future = client.moveOnPathAsync(
        waypoints,
        float(args.speed_m_s),
        timeout_sec=max(30.0, total_time * 2.0),
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
        vehicle_name=args.vehicle,
    )

    rows: List[Dict[str, Any]] = []
    collision_count = 0
    capture_failures = 0
    t0 = time.time()
    for step in range(args.steps):
        fpv_path = frames_fpv / f"frame_{step:06d}.png"
        chase_path = frames_chase / f"frame_{step:06d}.png"
        if bool(args.overlay):
            requests = [
                airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
                airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
            ]
            responses = _sim_get_images_with_retry(client, requests, vehicle_name=args.vehicle)
            if responses is None:
                capture_failures += 1
                time.sleep(float(args.dt))
                continue
            state = client.getMultirotorState(vehicle_name=args.vehicle)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            dx = float(pos.x_val) - float(target_x)
            dy = float(pos.y_val) - float(target_y)
            dist = float((dx * dx + dy * dy) ** 0.5)
            speed = math.sqrt(float(vel.x_val) ** 2 + float(vel.y_val) ** 2 + float(vel.z_val) ** 2)
            collided = bool(getattr(client.simGetCollisionInfo(vehicle_name=args.vehicle), "has_collided", False))
            overlay = [
                OverlayLine("AUXLINE (AirSim physics)", (255, 255, 255), scale=1.05),
                OverlayLine(f"step={step}/{args.steps - 1}   t={float(step) * float(args.dt):.2f}s", (220, 220, 220)),
                OverlayLine(f"pos_ned=({float(pos.x_val):.1f},{float(pos.y_val):.1f},{float(pos.z_val):.1f})", (255, 255, 0)),
                OverlayLine(
                    f"vel_ned=({float(vel.x_val):+.1f},{float(vel.y_val):+.1f},{float(vel.z_val):+.1f}) m/s   speed={float(speed):.1f}",
                    (0, 255, 0),
                ),
                OverlayLine(f"dist_to_target={dist:.1f} m", (0, 255, 255)),
                OverlayLine(f"target_ned=({float(target_x):.1f},{float(target_y):.1f},0.0)", (200, 200, 200)),
                OverlayLine(
                    "ENV: "
                    f"time={args.time_of_day or 'default'}  "
                    f"fog={args.weather_fog if args.weather_fog is not None else 0.0:.2f}  "
                    f"rain={args.weather_rain if args.weather_rain is not None else 0.0:.2f}  "
                    f"snow={args.weather_snow if args.weather_snow is not None else 0.0:.2f}  "
                    f"roadSnow={args.weather_road_snow if args.weather_road_snow is not None else 0.0:.2f}",
                    (200, 200, 200),
                ),
                OverlayLine(
                    f"wind_ned=({float(args.wind_x):+.1f},{float(args.wind_y):+.1f},{float(args.wind_z):+.1f})",
                    (200, 200, 200),
                ),
                OverlayLine(f"collision={'YES' if collided else 'no'}", (0, 0, 255) if collided else (200, 200, 200)),
            ]
            _write_png_with_overlay(fpv_path, responses[0].image_data_uint8, overlay)
            _write_png_with_overlay(chase_path, responses[1].image_data_uint8, overlay)
        else:
            _capture_dual_rgb(airsim, client, vehicle_name=args.vehicle, fpv_path=fpv_path, chase_path=chase_path)

        collision = client.simGetCollisionInfo(vehicle_name=args.vehicle)
        if getattr(collision, "has_collided", False):
            collision_count += 1

        state = client.getMultirotorState(vehicle_name=args.vehicle)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        rows.append(
            {
                "run_id": ctx.run_id,
                "mode": "auxline",
                "step": step,
                "time_s": float(step) * float(args.dt),
                "x_ned": float(pos.x_val),
                "y_ned": float(pos.y_val),
                "z_ned": float(pos.z_val),
                "vx_ned": float(vel.x_val),
                "vy_ned": float(vel.y_val),
                "vz_ned": float(vel.z_val),
                "target_x_ned": float(target_x),
                "target_y_ned": float(target_y),
                "wind_x": float(args.wind_x),
                "wind_y": float(args.wind_y),
                "wind_z": float(args.wind_z),
                "has_collided": bool(getattr(collision, "has_collided", False)),
            }
        )
        time.sleep(float(args.dt))

    runtime_s = time.time() - t0

    try:
        path_future.join()
    except Exception:
        pass

    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir / "timeseries.parquet", index=False)

    video_path = artifacts_dir / "video.mp4"
    frames_written, width_px = build_split_video(
        frames_left_dir=frames_fpv,
        frames_right_dir=frames_chase,
        output_path=video_path,
        fps=args.fps,
    )

    summary = {
        "run_id": ctx.run_id,
        "mode": "auxline",
        "steps": int(args.steps),
        "dt": float(args.dt),
        "fps": float(args.fps),
        "runtime_s": float(runtime_s),
        "collision_count": int(collision_count),
        "capture_failures": int(capture_failures),
        "wind_ned": [float(args.wind_x), float(args.wind_y), float(args.wind_z)],
        "video": {"path": str(video_path), "frames": int(frames_written), "width_px": int(width_px)},
    }
    write_json(ctx.run_dir / "summary.json", summary)
    (ctx.run_dir / "summary.txt").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Done:", ctx.run_dir)
    print("Video:", video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
