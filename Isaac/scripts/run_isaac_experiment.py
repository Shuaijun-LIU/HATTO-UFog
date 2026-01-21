from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.runpack import prepare_run, write_json


def _rpm_to_rad_s(rpm: float) -> float:
    return float(rpm) * 2.0 * math.pi / 60.0


def _quat_to_euler_wxyz(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=float,
    )


def _wrap_pi(angle_rad: float) -> float:
    x = float(angle_rad)
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_save_rgb(path: Path, rgb_u8: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb_u8, mode="RGB").save(str(path))
        return True
    except Exception:
        pass
    try:
        import cv2

        bgr = rgb_u8[:, :, ::-1]
        return bool(cv2.imwrite(str(path), bgr))
    except Exception:
        # Isaac python env may not have Pillow/OpenCV. Fall back to a raw numpy dump
        # and let postprocess_video.py turn it into an MP4.
        try:
            npy_path = path.with_suffix(".npy")
            np.save(npy_path, rgb_u8)
            return True
        except Exception:
            return False


def _normalize_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    img = np.asarray(arr)
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={img.shape}")
    if img.shape[2] >= 3:
        img = img[:, :, :3]
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _try_import_camera_class():
    candidates = [
        ("isaacsim.core.api.objects", "Camera"),
        ("isaacsim.sensors.camera", "Camera"),
        ("omni.isaac.sensor", "Camera"),
    ]
    for module_name, attr in candidates:
        try:
            mod = __import__(module_name, fromlist=[attr])
            return getattr(mod, attr)
        except Exception:
            continue
    return None


def _mix_to_omegas(
    *,
    total_thrust_n: float,
    body_torque_n_m: np.ndarray,
    kf: float,
    km: float,
    arm_length_m: float,
    omega_min: float,
    omega_max: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map (T, tau) -> rotor omegas (rad/s), using a simple PLUS layout.

    Returns (omegas, debug_dict).
    """

    total_thrust_n = float(max(0.0, total_thrust_n))
    tau = np.asarray(body_torque_n_m, dtype=float).reshape(3)

    if kf <= 0.0:
        raise ValueError("kf must be > 0")
    if arm_length_m <= 0.0:
        raise ValueError("arm_length_m must be > 0")

    # thrusts per rotor (N)
    # PLUS layout:
    # - rotor 1: +x
    # - rotor 2: +y
    # - rotor 3: -x
    # - rotor 4: -y
    # tau_x = arm*(f2 - f4)
    # tau_y = arm*(f3 - f1)
    # tau_z = km*(f1 - f2 + f3 - f4)  (if km != 0)
    if abs(km) > 1e-12:
        A = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, arm_length_m, 0.0, -arm_length_m],
                [-arm_length_m, 0.0, arm_length_m, 0.0],
                [km, -km, km, -km],
            ],
            dtype=float,
        )
        b = np.array([total_thrust_n, float(tau[0]), float(tau[1]), float(tau[2])], dtype=float)
        f, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        A = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, arm_length_m, 0.0, -arm_length_m],
                [-arm_length_m, 0.0, arm_length_m, 0.0],
            ],
            dtype=float,
        )
        b = np.array([total_thrust_n, float(tau[0]), float(tau[1])], dtype=float)
        f, *_ = np.linalg.lstsq(A, b, rcond=None)
        # yaw torque not representable when km==0; set to 0.
        tau = np.array([float(tau[0]), float(tau[1]), 0.0], dtype=float)

    f = np.maximum(0.0, f)
    omegas = np.sqrt(np.maximum(0.0, f / kf))
    omegas = np.clip(omegas, float(omega_min), float(omega_max)).astype(np.float32)
    f_applied = (kf * (omegas.astype(float) ** 2)).astype(float)

    tau_x = arm_length_m * (f_applied[1] - f_applied[3])
    tau_y = arm_length_m * (f_applied[2] - f_applied[0])
    tau_z = km * (f_applied[0] - f_applied[1] + f_applied[2] - f_applied[3]) if abs(km) > 1e-12 else 0.0

    dbg = {
        "f_cmd": f.tolist(),
        "f_applied": f_applied.tolist(),
        "tau_applied": [float(tau_x), float(tau_y), float(tau_z)],
        "thrust_applied": float(np.sum(f_applied)),
    }
    return omegas, dbg


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac Sim main runner (minimal open-source baseline).")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output", default="runs_isaac", help="Output root folder under Isaac/")
    parser.add_argument("--name", default="hover_smoke", help="Run name (used for run_id + slug).")
    parser.add_argument("--steps", type=int, default=None, help="Override sim.steps from config (for smoke tests).")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    output_root = (Path(__file__).resolve().parents[1] / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run = prepare_run(output_root=output_root, name=args.name, extra_meta={"config_path": str(cfg_path)})
    write_json(run.run_dir / "config.json", cfg)

    sim_cfg = cfg.get("sim", {})
    asset_cfg = cfg.get("asset", {})
    uav_cfg = cfg.get("uav", {})
    dist_cfg = cfg.get("disturbance", {})
    log_cfg = cfg.get("logging", {})

    ctrl_cfg = cfg.get("control", {})
    control_mode = str(ctrl_cfg.get("mode", "open_loop_hover")).strip()
    targets_cfg = ctrl_cfg.get("targets", {})
    target_z_m = float(targets_cfg.get("z_m", 1.0))
    target_roll_rad = float(targets_cfg.get("roll_rad", 0.0))
    target_pitch_rad = float(targets_cfg.get("pitch_rad", 0.0))
    target_yaw_rad = float(targets_cfg.get("yaw_rad", 0.0))

    pid_cfg = ctrl_cfg.get("pid", {})
    pid_z_cfg = pid_cfg.get("z", {})
    pid_att_cfg = pid_cfg.get("att", {})
    kp_z = float(pid_z_cfg.get("kp", 8.0))
    ki_z = float(pid_z_cfg.get("ki", 0.0))
    kd_z = float(pid_z_cfg.get("kd", 4.0))
    int_z_limit = float(pid_z_cfg.get("integrator_limit", 5.0))

    kp_att = np.array(pid_att_cfg.get("kp", [15.0, 15.0, 8.0]), dtype=float).reshape(3)
    kd_att = np.array(pid_att_cfg.get("kd", [3.0, 3.0, 2.0]), dtype=float).reshape(3)

    headless = bool(sim_cfg.get("headless", True))
    physics_dt_s = float(sim_cfg.get("physics_dt_s", 0.01))
    rendering_dt_s = float(sim_cfg.get("rendering_dt_s", physics_dt_s))
    steps = int(sim_cfg.get("steps", 600))
    if args.steps is not None:
        steps = int(args.steps)

    uav_body_prim_path = str(uav_cfg.get("body_prim_path", "/World/UAV/body"))
    mass_kg = float(uav_cfg.get("mass_kg", 1.8))
    inertia_xx = float(uav_cfg.get("inertia_xx", 0.03))
    inertia_yy = float(uav_cfg.get("inertia_yy", 0.03))
    inertia_zz = float(uav_cfg.get("inertia_zz", 0.06))
    gravity = float(uav_cfg.get("gravity_m_s2", 9.81))
    kf = float(uav_cfg.get("kf", 2e-5))
    km = float(uav_cfg.get("km", 0.0))
    arm_length_m = float(uav_cfg.get("arm_length_m", 0.25))

    omega_unit = str(uav_cfg.get("omega_unit", "rpm"))
    omega_min = float(uav_cfg.get("omega_min", 200.0))
    omega_max = float(uav_cfg.get("omega_max", 5000.0))
    if omega_unit == "rpm":
        omega_min = _rpm_to_rad_s(omega_min)
        omega_max = _rpm_to_rad_s(omega_max)

    wind_cfg = dist_cfg.get("wind", {})
    wind_enabled = bool(wind_cfg.get("enabled", False))
    wind_base = np.array(wind_cfg.get("base_vel_m_s", wind_cfg.get("wind_vel_m_s", [0.0, 0.0, 0.0])), dtype=float)
    wind_gust_amp = np.array(wind_cfg.get("gust_amp_m_s", [0.0, 0.0, 0.0]), dtype=float)
    wind_gust_freq_hz = float(wind_cfg.get("gust_freq_hz", 0.0))
    wind_noise_std = np.array(wind_cfg.get("noise_std_m_s", [0.0, 0.0, 0.0]), dtype=float)
    wind_ou_sigma = np.array(wind_cfg.get("ou_sigma_m_s", [0.0, 0.0, 0.0]), dtype=float)
    wind_ou_tau_s = float(wind_cfg.get("ou_tau_s", 0.0))
    wind_accel_gain = float(wind_cfg.get("accel_gain", 0.0))

    latency_cfg = dist_cfg.get("latency", {})
    latency_enabled = bool(latency_cfg.get("enabled", False))
    latency_ms = float(latency_cfg.get("latency_ms", 0.0)) if latency_enabled else 0.0
    latency_steps = int(round(max(0.0, latency_ms) / 1000.0 / max(1e-9, physics_dt_s))) if latency_enabled else 0

    vib_cfg = dist_cfg.get("vibration", {})
    torque_noise_std = float(vib_cfg.get("torque_noise_std", 0.0)) if bool(vib_cfg.get("enabled", False)) else 0.0

    meas_cfg = dist_cfg.get("measurement_noise", {})
    meas_enabled = bool(meas_cfg.get("enabled", False))
    meas_pos_std_m = float(meas_cfg.get("pos_std_m", 0.0)) if meas_enabled else 0.0
    meas_vel_std_m_s = float(meas_cfg.get("vel_std_m_s", 0.0)) if meas_enabled else 0.0
    meas_angle_std_rad = float(meas_cfg.get("angle_std_rad", 0.0)) if meas_enabled else 0.0
    meas_angvel_std_rad_s = float(meas_cfg.get("angvel_std_rad_s", 0.0)) if meas_enabled else 0.0

    capture_cfg = cfg.get("capture", {})
    capture_enabled = bool(capture_cfg.get("enabled", False))
    capture_every_n = int(capture_cfg.get("every_n_steps", 1)) if capture_enabled else 0
    capture_every_n = max(1, capture_every_n) if capture_enabled else 0
    capture_res = capture_cfg.get("resolution", [640, 360])
    try:
        capture_res = (int(capture_res[0]), int(capture_res[1]))
    except Exception:
        capture_res = (640, 360)
    capture_strict = bool(capture_cfg.get("strict", False))

    timeseries_name = str(log_cfg.get("timeseries_name", "timeseries.jsonl"))
    timeseries_path = run.run_dir / timeseries_name

    # Start Isaac Sim (slow).
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import RigidPrim

        world = World(
            physics_dt=physics_dt_s,
            rendering_dt=rendering_dt_s,
            stage_units_in_meters=1.0,
            backend="numpy",
        )
        # Open-source friendly: avoid `add_default_ground_plane()` because it may depend on external asset packs.
        FixedCuboid(
            prim_path="/World/ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.05], dtype=float),
            scale=np.array([200.0, 200.0, 0.1], dtype=float),
            size=1.0,
        )

        asset_mode = str(asset_cfg.get("mode", "primitive")).strip()
        if asset_mode == "nvidia_quadcopter":
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.storage.native import get_assets_root_path

            assets_root = get_assets_root_path()
            rel = str(asset_cfg.get("nvidia_relative_usd", "")).strip()
            if not assets_root or not rel:
                raise RuntimeError("Missing assets root or nvidia_relative_usd in config.")
            add_reference_to_stage(assets_root + rel, uav_cfg.get("prim_path", "/World/UAV"))
        elif asset_mode == "primitive":
            # Simple rigid body representing the UAV.
            DynamicCuboid(
                prim_path=uav_body_prim_path,
                name="uav_body",
                position=np.array([0.0, 0.0, 1.0], dtype=float),
                scale=np.array([0.4, 0.4, 0.1], dtype=float),
                mass=mass_kg,
            )
        else:
            raise RuntimeError(f"Unknown asset.mode: {asset_mode}")

        uav_view = RigidPrim(prim_paths_expr=uav_body_prim_path, name="uav_view")
        world.scene.add(uav_view)
        world.reset()
        world.play()
        uav_view.initialize()

        artifacts_dir = run.run_dir / "artifacts"
        frames_fpv_dir = artifacts_dir / "frames_fpv"
        frames_chase_dir = artifacts_dir / "frames_chase"

        fpv_cam = None
        chase_cam = None
        camera_cls = None
        if capture_enabled:
            camera_cls = _try_import_camera_class()
            if camera_cls is None:
                msg = "Capture requested but Camera class not found (try enabling isaacsim/omni camera extensions)."
                if capture_strict:
                    raise RuntimeError(msg)
                print("[warn]", msg)
                capture_enabled = False
            else:
                try:
                    fpv_cam = camera_cls(
                        prim_path=f"{uav_body_prim_path}/fpv_camera",
                        name="fpv_camera",
                        position=np.array([0.4, 0.0, 0.15], dtype=float),
                        resolution=capture_res,
                    )
                    chase_cam = camera_cls(
                        prim_path="/World/chase_camera",
                        name="chase_camera",
                        position=np.array([-6.0, -6.0, 3.0], dtype=float),
                        resolution=capture_res,
                    )
                    try:
                        world.scene.add(fpv_cam)
                        world.scene.add(chase_cam)
                    except Exception:
                        pass
                    try:
                        fpv_cam.initialize()
                        chase_cam.initialize()
                    except Exception:
                        pass
                    frames_fpv_dir.mkdir(parents=True, exist_ok=True)
                    frames_chase_dir.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    msg = f"Capture init failed ({type(exc).__name__}: {exc})"
                    if capture_strict:
                        raise RuntimeError(msg)
                    print("[warn]", msg)
                    capture_enabled = False
                    fpv_cam = None
                    chase_cam = None

        # Try to set inertia (best-effort; depends on physics handle readiness).
        inertia = np.array([[inertia_xx, 0.0, 0.0, 0.0, inertia_yy, 0.0, 0.0, 0.0, inertia_zz]], dtype=np.float32)
        try:
            uav_view.set_inertias(inertia)
        except Exception:
            pass

        # Open-loop hover: equal rotor speeds that approximate mg.
        # thrust = kf * omega^2, total = 4 * thrust
        omega_hover = math.sqrt(max(1e-9, mass_kg * gravity / (4.0 * max(kf, 1e-12))))
        omega_hover = float(np.clip(omega_hover, omega_min, omega_max))
        omega_cmd = np.array([omega_hover, omega_hover, omega_hover, omega_hover], dtype=np.float32)

        rng = np.random.default_rng(int(sim_cfg.get("seed", 0)))
        ou_state = np.zeros(3, dtype=float)
        omega_buffer = deque([omega_cmd.copy()] * (latency_steps + 1)) if latency_steps > 0 else None
        int_z = 0.0

        max_abs_roll = 0.0
        max_abs_pitch = 0.0
        for step_idx in range(steps):
            pos, quat = uav_view.get_world_poses()
            vel6 = uav_view.get_velocities()
            p = np.asarray(pos[0], dtype=float)
            q = np.asarray(quat[0], dtype=float)
            v = np.asarray(vel6[0, 0:3], dtype=float)
            w = np.asarray(vel6[0, 3:6], dtype=float)
            roll, pitch, yaw = _quat_to_euler_wxyz(q)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))

            rot = _quat_wxyz_to_rotmat(q)
            w_body = rot.T @ w

            # Optional measurement noise (feeds control only; log both true/measured).
            p_meas = p.copy()
            v_meas = v.copy()
            rpy_meas = np.array([roll, pitch, yaw], dtype=float)
            w_body_meas = w_body.copy()
            if meas_enabled:
                if meas_pos_std_m > 0:
                    p_meas = p_meas + rng.normal(0.0, meas_pos_std_m, size=3)
                if meas_vel_std_m_s > 0:
                    v_meas = v_meas + rng.normal(0.0, meas_vel_std_m_s, size=3)
                if meas_angle_std_rad > 0:
                    rpy_meas = rpy_meas + rng.normal(0.0, meas_angle_std_rad, size=3)
                if meas_angvel_std_rad_s > 0:
                    w_body_meas = w_body_meas + rng.normal(0.0, meas_angvel_std_rad_s, size=3)

            # Control (omega_cmd, body_force/torque are derived later).
            torque_cmd = np.zeros(3, dtype=float)
            total_thrust_cmd = 4.0 * kf * float(omega_hover * omega_hover)
            if control_mode == "open_loop_hover":
                omega_cmd = np.array([omega_hover, omega_hover, omega_hover, omega_hover], dtype=np.float32)
            elif control_mode == "pid_hover":
                err_z = float(target_z_m - p_meas[2])
                err_vz = float(0.0 - v_meas[2])
                int_z = float(np.clip(int_z + err_z * physics_dt_s, -int_z_limit, int_z_limit))
                a_z = kp_z * err_z + kd_z * err_vz + ki_z * int_z
                total_thrust_cmd = mass_kg * max(0.0, gravity + a_z)

                err_roll = float(target_roll_rad - rpy_meas[0])
                err_pitch = float(target_pitch_rad - rpy_meas[1])
                err_yaw = _wrap_pi(float(target_yaw_rad - rpy_meas[2]))
                err_rpy = np.array([err_roll, err_pitch, err_yaw], dtype=float)
                torque_cmd = kp_att * err_rpy - kd_att * w_body_meas

                omega_cmd, _ = _mix_to_omegas(
                    total_thrust_n=total_thrust_cmd,
                    body_torque_n_m=torque_cmd,
                    kf=kf,
                    km=km,
                    arm_length_m=arm_length_m,
                    omega_min=omega_min,
                    omega_max=omega_max,
                )
            else:
                raise RuntimeError(f"Unknown control.mode: {control_mode}")

            # Apply actuator latency (omega_cmd -> omega_applied).
            if omega_buffer is not None:
                omega_buffer.append(omega_cmd.copy())
                omega_applied = omega_buffer.popleft()
            else:
                omega_applied = omega_cmd

            thrusts = kf * (omega_applied.astype(float) ** 2)
            total_thrust = float(np.sum(thrusts))

            # Body torque derived from applied omegas (PLUS layout; see _mix_to_omegas).
            tau_x = arm_length_m * (thrusts[1] - thrusts[3])
            tau_y = arm_length_m * (thrusts[2] - thrusts[0])
            tau_z = km * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3]) if abs(km) > 1e-12 else 0.0
            body_force = np.array([0.0, 0.0, total_thrust], dtype=np.float32).reshape(1, 3)
            body_torque = np.array([tau_x, tau_y, tau_z], dtype=np.float32).reshape(1, 3)

            if torque_noise_std > 0:
                body_torque = body_torque + rng.normal(0.0, torque_noise_std, size=(1, 3)).astype(np.float32)

            uav_view.apply_forces_and_torques_at_pos(forces=body_force, torques=body_torque, is_global=False)

            wind_vel = wind_base.copy()
            if wind_enabled:
                if wind_gust_freq_hz > 0.0 and float(np.linalg.norm(wind_gust_amp)) > 0.0:
                    wind_vel = wind_vel + wind_gust_amp * math.sin(2.0 * math.pi * wind_gust_freq_hz * (step_idx * physics_dt_s))
                if float(np.linalg.norm(wind_ou_sigma)) > 0.0 and wind_ou_tau_s > 0.0:
                    ou_state = ou_state + (-(ou_state) / wind_ou_tau_s) * physics_dt_s + wind_ou_sigma * math.sqrt(
                        max(0.0, physics_dt_s)
                    ) * rng.normal(0.0, 1.0, size=3)
                    wind_vel = wind_vel + ou_state
                if float(np.linalg.norm(wind_noise_std)) > 0.0:
                    wind_vel = wind_vel + rng.normal(0.0, wind_noise_std, size=3)

            if wind_enabled and wind_accel_gain > 0:
                wind_force_world = (wind_accel_gain * (wind_vel - v) * mass_kg).astype(np.float32).reshape(1, 3)
                uav_view.apply_forces(wind_force_world, is_global=True)

            world.step(render=(not headless) or capture_enabled)

            if capture_enabled and (step_idx % capture_every_n == 0) and fpv_cam is not None and chase_cam is not None:
                try:
                    fpv = _normalize_rgb_uint8(fpv_cam.get_rgba())
                    chase = _normalize_rgb_uint8(chase_cam.get_rgba())
                    _maybe_save_rgb(frames_fpv_dir / f"frame_{step_idx:06d}.png", fpv)
                    _maybe_save_rgb(frames_chase_dir / f"frame_{step_idx:06d}.png", chase)
                except Exception as exc:
                    msg = f"Capture failed at step {step_idx} ({type(exc).__name__}: {exc})"
                    if capture_strict:
                        raise RuntimeError(msg)
                    if step_idx == 0:
                        print("[warn]", msg)

            _write_jsonl(
                timeseries_path,
                {
                    "step": step_idx,
                    "t_s": step_idx * physics_dt_s,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "vx": float(v[0]),
                    "vy": float(v[1]),
                    "vz": float(v[2]),
                    "qw": float(q[0]),
                    "qx": float(q[1]),
                    "qy": float(q[2]),
                    "qz": float(q[3]),
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll_meas": float(rpy_meas[0]),
                    "pitch_meas": float(rpy_meas[1]),
                    "yaw_meas": float(rpy_meas[2]),
                    "omega_x": float(w[0]),
                    "omega_y": float(w[1]),
                    "omega_z": float(w[2]),
                    "omega_body_x": float(w_body[0]),
                    "omega_body_y": float(w_body[1]),
                    "omega_body_z": float(w_body[2]),
                    "omega_body_meas_x": float(w_body_meas[0]),
                    "omega_body_meas_y": float(w_body_meas[1]),
                    "omega_body_meas_z": float(w_body_meas[2]),
                    "rotor_omega_cmd_1": float(omega_cmd[0]),
                    "rotor_omega_cmd_2": float(omega_cmd[1]),
                    "rotor_omega_cmd_3": float(omega_cmd[2]),
                    "rotor_omega_cmd_4": float(omega_cmd[3]),
                    "rotor_omega_1": float(omega_applied[0]),
                    "rotor_omega_2": float(omega_applied[1]),
                    "rotor_omega_3": float(omega_applied[2]),
                    "rotor_omega_4": float(omega_applied[3]),
                    "thrust_total_n": float(total_thrust),
                    "tau_x_n_m": float(tau_x),
                    "tau_y_n_m": float(tau_y),
                    "tau_z_n_m": float(tau_z),
                    "latency_ms": float(latency_ms) if latency_enabled else 0.0,
                    "latency_steps": int(latency_steps) if latency_enabled else 0,
                    "wind_x": float(wind_vel[0]) if wind_enabled else 0.0,
                    "wind_y": float(wind_vel[1]) if wind_enabled else 0.0,
                    "wind_z": float(wind_vel[2]) if wind_enabled else 0.0,
                    "control_mode": str(control_mode),
                },
            )

        summary = {
            "steps": steps,
            "physics_dt_s": physics_dt_s,
            "omega_hover": omega_hover,
            "max_abs_roll_rad": max_abs_roll,
            "max_abs_pitch_rad": max_abs_pitch,
            "control": {
                "mode": str(control_mode),
                "targets": {
                    "z_m": float(target_z_m),
                    "roll_rad": float(target_roll_rad),
                    "pitch_rad": float(target_pitch_rad),
                    "yaw_rad": float(target_yaw_rad),
                },
                "pid": {
                    "z": {"kp": float(kp_z), "ki": float(ki_z), "kd": float(kd_z), "integrator_limit": float(int_z_limit)},
                    "att": {"kp": kp_att.tolist(), "kd": kd_att.tolist()},
                },
            },
            "disturbance": {
                "wind_enabled": bool(wind_enabled),
                "wind_accel_gain": float(wind_accel_gain),
                "latency_enabled": bool(latency_enabled),
                "latency_ms": float(latency_ms),
                "latency_steps": int(latency_steps),
                "vibration_enabled": bool(vib_cfg.get("enabled", False)),
                "torque_noise_std": float(torque_noise_std),
                "measurement_noise_enabled": bool(meas_enabled),
                "measurement": {
                    "pos_std_m": float(meas_pos_std_m),
                    "vel_std_m_s": float(meas_vel_std_m_s),
                    "angle_std_rad": float(meas_angle_std_rad),
                    "angvel_std_rad_s": float(meas_angvel_std_rad_s),
                },
            },
            "capture": {
                "enabled": bool(capture_enabled),
                "every_n_steps": int(capture_every_n) if capture_enabled else 0,
                "resolution": list(capture_res) if capture_enabled else None,
                "frames_fpv_dir": str(frames_fpv_dir) if capture_enabled else None,
                "frames_chase_dir": str(frames_chase_dir) if capture_enabled else None,
            },
        }
        write_json(run.run_dir / str(log_cfg.get("summary_name", "summary.json")), summary)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
