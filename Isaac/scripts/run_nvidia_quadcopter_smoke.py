from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.runpack import prepare_run, write_json


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _prim_exists(stage, prim_path: str) -> bool:
    try:
        prim = stage.GetPrimAtPath(str(prim_path))
        return bool(prim and prim.IsValid())
    except Exception:
        return False


def _find_first_rigid_body_prim(stage, root_prim_path: str) -> Optional[str]:
    prefix = str(root_prim_path)
    try:
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not path.startswith(prefix):
                continue
            if prim.HasAttribute("physics:rigidBodyEnabled"):
                return path
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA quadcopter smoke (asset-based fallback; no custom controller).")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output", default="runs_isaac", help="Output root folder under Isaac/")
    parser.add_argument("--name", default="nvidia_quadcopter_smoke", help="Run name.")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    output_root = (Path(__file__).resolve().parents[1] / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run = prepare_run(output_root=output_root, name=args.name, extra_meta={"config_path": str(cfg_path), "mode": "nvidia"})
    write_json(run.run_dir / "config.json", cfg)

    sim_cfg = cfg.get("sim", {})
    asset_cfg = cfg.get("asset", {})
    uav_cfg = cfg.get("uav", {})
    log_cfg = cfg.get("logging", {})
    capture_cfg = cfg.get("capture", {})

    headless = bool(sim_cfg.get("headless", True))
    physics_dt_s = float(sim_cfg.get("physics_dt_s", 0.01))
    rendering_dt_s = float(sim_cfg.get("rendering_dt_s", physics_dt_s))
    steps = int(sim_cfg.get("steps", 240))

    uav_prim_path = str(uav_cfg.get("prim_path", "/World/UAV"))
    uav_body_prim_path = str(uav_cfg.get("body_prim_path", "")).strip()
    rel_usd = str(asset_cfg.get("nvidia_relative_usd", "/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd")).strip()

    capture_enabled = bool(capture_cfg.get("enabled", False))
    capture_every_n = int(capture_cfg.get("every_n_steps", 5)) if capture_enabled else 0
    capture_every_n = max(1, capture_every_n) if capture_enabled else 0
    capture_res = capture_cfg.get("resolution", [640, 360])
    try:
        capture_res = (int(capture_res[0]), int(capture_res[1]))
    except Exception:
        capture_res = (640, 360)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        world = World(
            physics_dt=physics_dt_s,
            rendering_dt=rendering_dt_s,
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

        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("get_assets_root_path() returned empty. Please set up Isaac Sim assets/Nucleus.")
        usd_path = assets_root + rel_usd
        add_reference_to_stage(usd_path, uav_prim_path)

        # Resolve a rigid body prim to track (best-effort).
        stage = None
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        except Exception:
            stage = None

        body_path = uav_body_prim_path if uav_body_prim_path else ""
        if stage is not None:
            if body_path and not _prim_exists(stage, body_path):
                body_path = ""
            if not body_path:
                found = _find_first_rigid_body_prim(stage, uav_prim_path)
                if found:
                    body_path = found

        if not body_path:
            # Last resort: try to track the root prim (may not be rigid).
            body_path = uav_prim_path

        uav_view = RigidPrim(prim_paths_expr=body_path, name="uav_view")
        world.scene.add(uav_view)
        world.reset()
        world.play()
        uav_view.initialize()

        artifacts_dir = run.run_dir / "artifacts"
        frames_dir = artifacts_dir / "frames"
        cam = None
        if capture_enabled:
            camera_cls = _try_import_camera_class()
            if camera_cls is not None:
                try:
                    cam = camera_cls(
                        prim_path="/World/smoke_camera",
                        name="smoke_camera",
                        position=np.array([-6.0, -6.0, 3.0], dtype=float),
                        resolution=capture_res,
                    )
                    try:
                        world.scene.add(cam)
                    except Exception:
                        pass
                    try:
                        cam.initialize()
                    except Exception:
                        pass
                    frames_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    cam = None

        timeseries_path = run.run_dir / str(log_cfg.get("timeseries_name", "timeseries.jsonl"))
        min_z = None
        for step_idx in range(steps):
            pos, quat = uav_view.get_world_poses()
            p = np.asarray(pos[0], dtype=float)
            min_z = float(p[2]) if min_z is None else float(min(min_z, float(p[2])))
            _write_jsonl(
                timeseries_path,
                {"step": step_idx, "t_s": float(step_idx) * float(physics_dt_s), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            )

            world.step(render=(not headless) or capture_enabled)
            if capture_enabled and cam is not None and (step_idx % capture_every_n == 0):
                try:
                    rgb = _normalize_rgb_uint8(cam.get_rgba())
                    _maybe_save_rgb(frames_dir / f"frame_{step_idx:06d}.png", rgb)
                except Exception:
                    pass

        summary = {
            "mode": "nvidia_quadcopter_smoke",
            "steps": int(steps),
            "physics_dt_s": float(physics_dt_s),
            "asset": {"usd_path": str(usd_path), "prim_path": str(uav_prim_path), "body_track_path": str(body_path)},
            "min_z_m": float(min_z) if min_z is not None else None,
            "capture": {
                "enabled": bool(capture_enabled and cam is not None),
                "frames_dir": str(frames_dir) if (capture_enabled and cam is not None) else None,
                "every_n_steps": int(capture_every_n) if (capture_enabled and cam is not None) else 0,
                "resolution": list(capture_res) if (capture_enabled and cam is not None) else None,
            },
        }
        write_json(run.run_dir / str(log_cfg.get("summary_name", "summary.json")), summary)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
