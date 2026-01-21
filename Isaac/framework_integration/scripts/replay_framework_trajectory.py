from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.runpack import prepare_run, write_json
from framework_integration.lib.trajectory_io import Trajectory, load_trajectory_json


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


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
        try:
            npy_path = path.with_suffix(".npy")
            np.save(npy_path, rgb_u8)
            return True
        except Exception:
            return False


def _teleport_best_effort(uav_view, *, pos: np.ndarray, quat_wxyz: np.ndarray, prim_path: str) -> None:
    # Try core API first.
    for name in ["set_world_poses", "set_world_pose"]:
        fn = getattr(uav_view, name, None)
        if callable(fn):
            try:
                if name == "set_world_poses":
                    fn(positions=pos.reshape(1, 3), orientations=quat_wxyz.reshape(1, 4))
                else:
                    fn(position=pos.reshape(3), orientation=quat_wxyz.reshape(4))
                return
            except Exception:
                break

    # Fall back to USD transform edit (works for visualization even if physics isn't authoritative).
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(str(prim_path))
        xform = UsdGeom.Xformable(prim)
        # Clear existing ops for determinism.
        xform.ClearXformOpOrder()
        t_op = xform.AddTranslateOp()
        q_op = xform.AddOrientOp()
        t_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        q_op.Set(Gf.Quatf(float(quat_wxyz[0]), Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))))
    except Exception as exc:
        raise RuntimeError(f"Failed to teleport UAV prim (no supported API): {type(exc).__name__}: {exc}") from exc


def _iter_steps(traj: Trajectory, *, dt_s: float, max_steps: int) -> Tuple[int, float]:
    n = len(traj.path_xyz)
    if max_steps and max_steps > 0:
        n = min(n, int(max_steps))
    duration = (n - 1) * float(dt_s) if n > 1 else 0.0
    return n, duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Framework trajectory.json in Isaac Sim (pose-driven; for visualization).")
    parser.add_argument("--trajectory_json", required=True, help="Path to trajectory.json (z-up, meters).")
    parser.add_argument("--output", default="runs_isaac", help="Output root folder under Isaac/")
    parser.add_argument("--name", default="fw_replay", help="Run name.")
    parser.add_argument("--headless", action="store_true", help="Run headless (still can capture frames).")
    parser.add_argument("--dt_s", type=float, default=0.05, help="Replay timestep (used if trajectory has no time_s).")
    parser.add_argument("--max_steps", type=int, default=0, help="If >0, cap replay steps.")

    parser.add_argument("--scale_xy", type=float, default=1.0)
    parser.add_argument("--scale_z", type=float, default=1.0)
    parser.add_argument("--x_offset_m", type=float, default=0.0)
    parser.add_argument("--y_offset_m", type=float, default=0.0)
    parser.add_argument("--z_offset_m", type=float, default=0.0)

    parser.add_argument("--capture", action="store_true", help="Capture frames (best-effort; no MP4 in Isaac env).")
    parser.add_argument("--capture_every_n_steps", type=int, default=5)
    parser.add_argument("--capture_res", nargs=2, type=int, default=[640, 360])
    parser.add_argument("--capture_strict", action="store_true", help="Fail if capture cannot be initialized.")
    args = parser.parse_args()

    traj_path = Path(args.trajectory_json).expanduser().resolve()
    traj = load_trajectory_json(traj_path)

    output_root = (Path(__file__).resolve().parents[2] / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run = prepare_run(
        output_root=output_root,
        name=args.name,
        extra_meta={"mode": "framework_replay", "trajectory_json": str(traj_path)},
    )

    write_json(
        run.run_dir / "config.json",
        {
            "trajectory_json": str(traj_path),
            "mapping": {
                "scale_xy": float(args.scale_xy),
                "scale_z": float(args.scale_z),
                "x_offset_m": float(args.x_offset_m),
                "y_offset_m": float(args.y_offset_m),
                "z_offset_m": float(args.z_offset_m),
            },
            "replay": {"dt_s": float(args.dt_s), "max_steps": int(args.max_steps), "headless": bool(args.headless)},
            "capture": {
                "enabled": bool(args.capture),
                "every_n_steps": int(args.capture_every_n_steps),
                "resolution": [int(args.capture_res[0]), int(args.capture_res[1])],
                "strict": bool(args.capture_strict),
            },
        },
    )

    timeseries_path = run.run_dir / "timeseries.jsonl"

    # Start Isaac Sim (slow).
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import RigidPrim

        dt_s = float(args.dt_s)
        world = World(
            physics_dt=dt_s,
            rendering_dt=dt_s,
            stage_units_in_meters=1.0,
            backend="numpy",
        )
        FixedCuboid(
            prim_path="/World/ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.05], dtype=float),
            scale=np.array([200.0, 200.0, 0.1], dtype=float),
            size=1.0,
        )

        uav_body_prim_path = "/World/UAV/body"
        DynamicCuboid(
            prim_path=uav_body_prim_path,
            name="uav_body",
            position=np.array([0.0, 0.0, 1.0], dtype=float),
            scale=np.array([0.4, 0.4, 0.1], dtype=float),
            mass=1.0,
        )
        uav_view = RigidPrim(prim_paths_expr=uav_body_prim_path, name="uav_view")
        world.scene.add(uav_view)
        world.reset()
        world.play()
        uav_view.initialize()

        capture_enabled = bool(args.capture)
        capture_every = max(1, int(args.capture_every_n_steps)) if capture_enabled else 0
        capture_res = (int(args.capture_res[0]), int(args.capture_res[1]))
        capture_strict = bool(args.capture_strict)

        # Use the same folder name as `Isaac/scripts/postprocess_video.py` expects for single-view capture.
        frames_dir = run.run_dir / "artifacts" / "frames"
        chase_cam = None
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
                    chase_cam = camera_cls(
                        prim_path="/World/chase_camera",
                        name="chase_camera",
                        position=np.array([-6.0, -6.0, 3.0], dtype=float),
                        resolution=capture_res,
                    )
                    try:
                        world.scene.add(chase_cam)
                    except Exception:
                        pass
                    try:
                        chase_cam.initialize()
                    except Exception:
                        pass
                    frames_dir.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    msg = f"Capture init failed ({type(exc).__name__}: {exc})"
                    if capture_strict:
                        raise RuntimeError(msg)
                    print("[warn]", msg)
                    capture_enabled = False
                    chase_cam = None

        n_steps, duration = _iter_steps(traj, dt_s=dt_s, max_steps=int(args.max_steps))
        for step_idx in range(n_steps):
            x, y, z = traj.path_xyz[step_idx]
            pos = np.array(
                [
                    float(args.scale_xy) * x + float(args.x_offset_m),
                    float(args.scale_xy) * y + float(args.y_offset_m),
                    float(args.scale_z) * z + float(args.z_offset_m),
                ],
                dtype=np.float32,
            )

            if traj.rpy_rad is not None:
                yaw = float(traj.rpy_rad[step_idx][2])
            elif traj.yaw_rad is not None:
                yaw = float(traj.yaw_rad[step_idx])
            else:
                yaw = 0.0
            quat_wxyz = _yaw_to_quat_wxyz(yaw)

            _teleport_best_effort(uav_view, pos=pos, quat_wxyz=quat_wxyz, prim_path=uav_body_prim_path)

            # Step simulation. Render if capture is enabled.
            world.step(render=bool(capture_enabled))

            # Best-effort state readback.
            pos_w, quat_w = uav_view.get_world_poses()
            p = np.asarray(pos_w[0], dtype=float)
            q = np.asarray(quat_w[0], dtype=float)

            row = {
                "step": int(step_idx),
                "time_s": float(step_idx * dt_s),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "quat_wxyz": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "x_cmd": float(pos[0]),
                "y_cmd": float(pos[1]),
                "z_cmd": float(pos[2]),
                "yaw_cmd_rad": float(yaw),
            }
            _write_jsonl(timeseries_path, row)

            if capture_enabled and chase_cam is not None and (step_idx % capture_every == 0):
                try:
                    rgba = chase_cam.get_rgba()
                    rgb = _normalize_rgb_uint8(np.asarray(rgba)[:, :, :3])
                    _maybe_save_rgb(frames_dir / f"frame_{step_idx:06d}.png", rgb)
                except Exception:
                    pass

        write_json(
            run.run_dir / "summary.json",
            {
                "steps": int(n_steps),
                "duration_s": float(duration),
                "capture_enabled": bool(capture_enabled),
                "frames_dir": str(frames_dir) if capture_enabled else None,
                "timeseries": str(timeseries_path),
            },
        )
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
