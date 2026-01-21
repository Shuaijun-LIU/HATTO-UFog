from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpackrpc
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.import_airsim import import_airsim
from bridge.overlay import OverlayLine, decode_png, draw_overlay_panel, encode_png
from bridge.paths import settings_json_path
from bridge.runpack import prepare_run, write_json
from bridge.video import build_split_video

from framework_integration.lib.transform import FwToNed, wrap_deg, yaw_rad_to_deg


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


def _required_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in timeseries: {missing}. Available={list(df.columns)}")


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _norm2(dx: float, dy: float) -> float:
    return float(math.sqrt(float(dx) * float(dx) + float(dy) * float(dy)))


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
        description="Track a Framework trajectory in AirSim auxline (AirSim physics) and log desired vs actual (feasibility validation)."
    )
    parser.add_argument("--framework_timeseries", required=True)
    parser.add_argument("--output_root", default="runs_airsim")
    parser.add_argument("--settings_template", default="configs/airsim_settings/settings_auxline.json")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout_s", type=int, default=30)
    parser.add_argument("--vehicle", default="Drone1")

    parser.add_argument("--dt", type=float, default=0.05, help="Command duration per step (and capture interval).")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--speed_m_s", type=float, default=3.0)

    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--z_col", default="z")
    parser.add_argument("--yaw_col", default="yaw")

    parser.add_argument("--scale_xy", type=float, default=1.0)
    parser.add_argument("--scale_z", type=float, default=1.0)
    parser.add_argument("--x_offset_ned", type=float, default=0.0)
    parser.add_argument("--y_offset_ned", type=float, default=0.0)
    parser.add_argument("--z_offset_ned", type=float, default=0.0)
    parser.add_argument("--x0", type=float, default=None, help="Optional: anchor X (NED). Defaults to current spawn X.")
    parser.add_argument("--y0", type=float, default=None, help="Optional: anchor Y (NED). Defaults to current spawn Y.")
    parser.add_argument("--z_up_m", type=float, default=None, help="Optional: force constant altitude (Framework z ignored).")
    parser.add_argument("--base_z_up_m", type=float, default=10.0, help="Anchor altitude when z_up_m is not provided.")

    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--plot_user_point", action="store_true")
    parser.add_argument("--user_label", default="USER")
    parser.add_argument("--yaw_mode", default="face_path", choices=["face_path", "from_timeseries", "fixed"])
    parser.add_argument("--fixed_yaw_deg", type=float, default=0.0)

    parser.add_argument("--dry_run", action="store_true", help="Do not connect to AirSim; only print planned info.")
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
    _required_columns(df, [args.x_col, args.y_col, args.z_col])
    has_yaw = args.yaw_col in df.columns

    n_total = len(df)
    if n_total == 0:
        raise ValueError("Empty timeseries.")
    n = n_total
    if args.max_steps and int(args.max_steps) > 0:
        n = min(n, int(args.max_steps))
        df = df.iloc[:n].copy()

    x_first = float(df.loc[0, args.x_col])
    y_first = float(df.loc[0, args.y_col])
    z_first = float(df.loc[0, args.z_col])

    # Prepare mapping; anchor is read from AirSim on connect (or set via x0/y0).
    mapper = FwToNed(
        scale_xy=float(args.scale_xy),
        scale_z=float(args.scale_z),
        x_offset_ned=float(args.x_offset_ned),
        y_offset_ned=float(args.y_offset_ned),
        z_offset_ned=float(args.z_offset_ned),
        z_up_m=float(args.z_up_m) if args.z_up_m is not None else None,
    )

    if bool(args.dry_run):
        print(f"[DRY-RUN] steps={n} dt={args.dt} fps={args.fps} speed_m_s={args.speed_m_s}")
        print(f"[DRY-RUN] mapping: {mapper}")
        return 0

    airsim = import_airsim()
    ctx = prepare_run(
        output_root=output_root,
        name="airsim_track_auxline",
        extra_meta={
            "mode": "track_auxline",
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
            "speed_m_s": float(args.speed_m_s),
            "mapping": mapper.__dict__,
            "yaw_mode": str(args.yaw_mode),
            "fixed_yaw_deg": float(args.fixed_yaw_deg),
        },
    )

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
        print("ERROR: Failed to connect to AirSim RPC server for track_auxline.")
        print("- Requires AirSim running with normal physics (NOT ExternalPhysicsEngine).")
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

    state0 = client.getMultirotorState(vehicle_name=args.vehicle)
    pos0 = state0.kinematics_estimated.position
    pose0 = client.simGetVehiclePose(vehicle_name=args.vehicle)

    x_anchor = float(args.x0) if args.x0 is not None else float(pos0.x_val)
    y_anchor = float(args.y0) if args.y0 is not None else float(pos0.y_val)
    z_anchor = -float(args.base_z_up_m)

    # Teleport to anchor at base altitude.
    try:
        pose0.position.x_val = float(x_anchor)
        pose0.position.y_val = float(y_anchor)
        pose0.position.z_val = float(z_anchor)
        pose0.orientation = airsim.to_quaternion(0.0, 0.0, 0.0)
        client.simSetVehiclePose(pose0, ignore_collision=True, vehicle_name=args.vehicle)
    except Exception:
        pass
    try:
        client.moveToZAsync(float(z_anchor), velocity=max(2.0, float(args.speed_m_s)), vehicle_name=args.vehicle).join()
    except Exception:
        pass

    # Precompute desired points in NED.
    desired: List[Tuple[float, float, float]] = []
    desired_yaw_deg: List[Optional[float]] = []
    for i in range(n):
        row = df.loc[i]
        x_fw = float(row[args.x_col])
        y_fw = float(row[args.y_col])
        z_fw = float(row[args.z_col])
        x_ned, y_ned, z_ned = mapper.map_point(
            x_fw=x_fw,
            y_fw=y_fw,
            z_fw=z_fw,
            x_first=x_first,
            y_first=y_first,
            z_first=z_first,
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            z_anchor=z_anchor,
        )
        desired.append((x_ned, y_ned, z_ned))
        if has_yaw:
            yaw_val = float(row[args.yaw_col])
            desired_yaw_deg.append(yaw_rad_to_deg(yaw_val) if math.isfinite(yaw_val) else None)
        else:
            desired_yaw_deg.append(None)

    # Choose a user/target marker (for visualization); default to first target if present.
    user_x_ned = x_anchor
    user_y_ned = y_anchor
    if "target_x" in df.columns and "target_y" in df.columns:
        tx_fw = float(df.loc[0, "target_x"])
        ty_fw = float(df.loc[0, "target_y"])
        tz_fw = float(df.loc[0, "target_z"]) if "target_z" in df.columns else z_first
        user_x_ned, user_y_ned, _user_z_ned = mapper.map_point(
            x_fw=tx_fw,
            y_fw=ty_fw,
            z_fw=tz_fw,
            x_first=x_first,
            y_first=y_first,
            z_first=z_first,
            x_anchor=x_anchor,
            y_anchor=y_anchor,
            z_anchor=z_anchor,
        )
    if bool(args.plot_user_point):
        try:
            p0 = airsim.Vector3r(float(user_x_ned), float(user_y_ned), 0.0)
            p1 = airsim.Vector3r(float(user_x_ned), float(user_y_ned), -30.0)
            client.simPlotLineList([p0, p1], color_rgba=[1.0, 0.2, 0.2, 1.0], thickness=10.0, duration=-1, is_persistent=True)
            client.simPlotPoints([p0], color_rgba=[1.0, 0.2, 0.2, 1.0], size=30.0, duration=-1, is_persistent=True)
            client.simPlotStrings([str(args.user_label)], [p1], scale=10, color_rgba=[1.0, 1.0, 0.2, 1.0], duration=-1)
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    capture_failures = 0
    collision_count = 0

    for step in range(n):
        row = df.loc[int(step)]
        x_des, y_des, z_des = desired[step]
        # Use current actual position for feedback velocity (pre-command).
        st0 = client.getMultirotorState(vehicle_name=args.vehicle)
        pos0 = st0.kinematics_estimated.position
        vel0 = st0.kinematics_estimated.linear_velocity
        x_act0 = float(pos0.x_val)
        y_act0 = float(pos0.y_val)
        z_act0 = float(pos0.z_val)

        dx = float(x_des) - float(x_act0)
        dy = float(y_des) - float(y_act0)
        dist_xy = _norm2(dx, dy)
        if float(args.dt) <= 1e-9:
            raise ValueError("--dt must be > 0")
        vx = dx / float(args.dt)
        vy = dy / float(args.dt)
        speed = _norm2(vx, vy)
        if speed > float(args.speed_m_s) and speed > 1e-9:
            s = float(args.speed_m_s) / float(speed)
            vx *= s
            vy *= s

        # Yaw control (AirSim expects degrees in yaw_mode).
        yaw_deg = float(args.fixed_yaw_deg)
        if args.yaw_mode == "face_path":
            if step < n - 1:
                x_next, y_next, _z_next = desired[min(step + 1, n - 1)]
                yaw_deg = yaw_rad_to_deg(math.atan2(float(y_next) - float(y_act0), float(x_next) - float(x_act0)))
            else:
                yaw_deg = yaw_rad_to_deg(math.atan2(float(user_y_ned) - float(y_act0), float(user_x_ned) - float(x_act0)))
        elif args.yaw_mode == "from_timeseries":
            if desired_yaw_deg[step] is not None:
                yaw_deg = float(desired_yaw_deg[step])
        yaw_deg = wrap_deg(yaw_deg)

        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=float(yaw_deg))
        try:
            client.moveByVelocityZAsync(
                float(vx),
                float(vy),
                float(z_des),
                duration=float(args.dt),
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=yaw_mode,
                vehicle_name=args.vehicle,
            ).join()
        except Exception:
            # Fail softly; still log state.
            pass

        # Post-command state (used for metrics/overlay/records).
        st = client.getMultirotorState(vehicle_name=args.vehicle)
        pos = st.kinematics_estimated.position
        vel = st.kinematics_estimated.linear_velocity
        x_act = float(pos.x_val)
        y_act = float(pos.y_val)
        z_act = float(pos.z_val)

        fpv_path = frames_fpv / f"frame_{step:06d}.png"
        chase_path = frames_chase / f"frame_{step:06d}.png"
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
            airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        ]
        responses = _sim_get_images_with_retry(client, requests, vehicle_name=args.vehicle) if bool(args.overlay) else client.simGetImages(requests, vehicle_name=args.vehicle)
        if responses is None:
            capture_failures += 1
            time.sleep(float(args.dt))
            continue

        collision = client.simGetCollisionInfo(vehicle_name=args.vehicle)
        has_collided = bool(getattr(collision, "has_collided", False))
        if has_collided:
            collision_count += 1

        err_xy = _norm2(float(x_act) - float(x_des), float(y_act) - float(y_des))
        err_z = float(z_act) - float(z_des)
        err_3d = float(math.sqrt(err_xy * err_xy + float(err_z) * float(err_z)))

        overlay: List[OverlayLine] = []
        if bool(args.overlay):
            overlay = [
                OverlayLine("AUXLINE TRACK (AirSim physics)", (255, 255, 255), scale=1.05),
                OverlayLine(f"step={step}/{n - 1}   t={float(step) * float(args.dt):.2f}s", (220, 220, 220)),
                OverlayLine(f"des_ned=({x_des:.1f},{y_des:.1f},{z_des:.1f})", (200, 200, 200)),
                OverlayLine(f"act_ned=({x_act:.1f},{y_act:.1f},{z_act:.1f})", (255, 255, 0)),
                OverlayLine(f"err_xy={err_xy:.2f} m  err_3d={err_3d:.2f} m", (0, 255, 255)),
                OverlayLine(f"cmd_vxy=({vx:+.2f},{vy:+.2f}) m/s  yaw={yaw_deg:+.1f} deg", (0, 255, 0)),
                OverlayLine(f"collision={'YES' if has_collided else 'no'}", (0, 0, 255) if has_collided else (200, 200, 200)),
            ]
            for key, label in [
                ("S", "S"),
                ("E_total", "E_total"),
                ("D_total", "D_total"),
                ("E_mov", "E_mov"),
                ("D_tr", "D_tr"),
                ("D_q", "D_q"),
                ("D_uavq", "D_uavq"),
            ]:
                s = _fmt_opt(row, key, "{:.3g}")
                if s is not None:
                    overlay.append(OverlayLine(f"{label}={s}", (200, 200, 200)))
        fpv_bytes = responses[0].image_data_uint8
        chase_bytes = responses[1].image_data_uint8
        _write_png_with_overlay(fpv_path, fpv_bytes, overlay)
        _write_png_with_overlay(chase_path, chase_bytes, overlay)

        rows.append(
            {
                "run_id": ctx.run_id,
                "mode": "track_auxline",
                "step": int(step),
                "time_s": float(step) * float(args.dt),
                "x_des_ned": float(x_des),
                "y_des_ned": float(y_des),
                "z_des_ned": float(z_des),
                "x_pre_ned": float(x_act0),
                "y_pre_ned": float(y_act0),
                "z_pre_ned": float(z_act0),
                "x_act_ned": float(x_act),
                "y_act_ned": float(y_act),
                "z_act_ned": float(z_act),
                "vx_pre_ned": float(vel0.x_val),
                "vy_pre_ned": float(vel0.y_val),
                "vz_pre_ned": float(vel0.z_val),
                "vx_act_ned": float(vel.x_val),
                "vy_act_ned": float(vel.y_val),
                "vz_act_ned": float(vel.z_val),
                "vx_cmd": float(vx),
                "vy_cmd": float(vy),
                "yaw_cmd_deg": float(yaw_deg),
                "err_xy_m": float(err_xy),
                "err_z_m": float(err_z),
                "err_3d_m": float(err_3d),
                "has_collided": bool(has_collided),
                **_metric_payload(row),
            }
        )
        time.sleep(float(args.dt))

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(ctx.run_dir / "timeseries.parquet", index=False)
    video_path = artifacts_dir / "video.mp4"
    frames_written, width_px = build_split_video(
        frames_left_dir=frames_fpv,
        frames_right_dir=frames_chase,
        output_path=video_path,
        fps=args.fps,
    )
    write_json(
        ctx.run_dir / "summary.json",
        {
            "run_id": ctx.run_id,
            "mode": "track_auxline",
            "steps": int(n),
            "dt": float(args.dt),
            "fps": float(args.fps),
            "capture_failures": int(capture_failures),
            "collision_count": int(collision_count),
            "frames_written": int(frames_written),
            "video_width_px": int(width_px),
        },
    )
    print(f"Output: {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
