from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    wind_vel = np.array(wind_cfg.get("wind_vel_m_s", [0.0, 0.0, 0.0]), dtype=float)
    wind_accel_gain = float(wind_cfg.get("accel_gain", 0.0))

    vib_cfg = dist_cfg.get("vibration", {})
    torque_noise_std = float(vib_cfg.get("torque_noise_std", 0.0)) if bool(vib_cfg.get("enabled", False)) else 0.0

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
        omegas = np.array([omega_hover, omega_hover, omega_hover, omega_hover], dtype=np.float32)

        rng = np.random.default_rng(int(sim_cfg.get("seed", 0)))

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

            # Net thrust/torque in body frame (very simplified).
            thrust_per = kf * float(omega_hover * omega_hover)
            total_thrust = 4.0 * thrust_per
            body_force = np.array([0.0, 0.0, total_thrust], dtype=np.float32).reshape(1, 3)

            tau_z = km * float(omegas[0] ** 2 - omegas[1] ** 2 + omegas[2] ** 2 - omegas[3] ** 2)
            body_torque = np.array([0.0, 0.0, tau_z], dtype=np.float32).reshape(1, 3)

            if torque_noise_std > 0:
                body_torque = body_torque + rng.normal(0.0, torque_noise_std, size=(1, 3)).astype(np.float32)

            uav_view.apply_forces_and_torques_at_pos(forces=body_force, torques=body_torque, is_global=False)

            if wind_enabled and wind_accel_gain > 0:
                wind_force_world = (wind_accel_gain * (wind_vel - v) * mass_kg).astype(np.float32).reshape(1, 3)
                uav_view.apply_forces(wind_force_world, is_global=True)

            world.step(render=not headless)

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
                    "omega_x": float(w[0]),
                    "omega_y": float(w[1]),
                    "omega_z": float(w[2]),
                    "rotor_omega_1": float(omegas[0]),
                    "rotor_omega_2": float(omegas[1]),
                    "rotor_omega_3": float(omegas[2]),
                    "rotor_omega_4": float(omegas[3]),
                    "wind_x": float(wind_vel[0]) if wind_enabled else 0.0,
                    "wind_y": float(wind_vel[1]) if wind_enabled else 0.0,
                    "wind_z": float(wind_vel[2]) if wind_enabled else 0.0,
                },
            )

        summary = {
            "steps": steps,
            "physics_dt_s": physics_dt_s,
            "omega_hover": omega_hover,
            "max_abs_roll_rad": max_abs_roll,
            "max_abs_pitch_rad": max_abs_pitch,
        }
        write_json(run.run_dir / str(log_cfg.get("summary_name", "summary.json")), summary)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
