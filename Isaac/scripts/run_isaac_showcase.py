from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.runpack import prepare_run, write_json


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        raise ValueError("Camera.get_rgba() returned None.")
    img = np.asarray(arr)
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={img.shape}")
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"Empty image returned by camera (shape={img.shape})")
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

        # Pillow 13 deprecates the explicit `mode` parameter; infer from array.
        Image.fromarray(rgb_u8).save(str(path))
        return True
    except Exception:
        pass
    try:
        import cv2

        bgr = rgb_u8[:, :, ::-1]
        return bool(cv2.imwrite(str(path), bgr))
    except Exception:
        try:
            np.save(path.with_suffix(".npy"), rgb_u8)
            return True
        except Exception:
            return False


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=float)
    q = q / max(1e-12, float(np.linalg.norm(q)))
    return q


def _look_at_quat_wxyz(camera_pos: np.ndarray, target_pos: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a USD-compatible look-at orientation.

    Prefer a USD-native implementation (`Gf.Matrix4d().SetLookAt(...)`) to avoid axis/sign
    mistakes across different kit/camera conventions.
    """
    cam = np.asarray(camera_pos, dtype=float).reshape(3)
    tgt = np.asarray(target_pos, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)

    try:
        from pxr import Gf

        eye = Gf.Vec3d(float(cam[0]), float(cam[1]), float(cam[2]))
        target = Gf.Vec3d(float(tgt[0]), float(tgt[1]), float(tgt[2]))
        up_axis = Gf.Vec3d(float(up[0]), float(up[1]), float(up[2]))

        # Handle collinearity between direction and up axis (same logic as Isaac replicator behavior).
        direction = target - eye
        cross = direction.GetCross(up_axis)
        if cross.GetLength() < 1e-5:
            if abs(up_axis[0]) < 1e-5:
                eye = eye + Gf.Vec3d(1, 0, 0) * 1e-5
            elif abs(up_axis[1]) < 1e-5:
                eye = eye + Gf.Vec3d(0, 1, 0) * 1e-5
            else:
                eye = eye + Gf.Vec3d(0, 0, 1) * 1e-5

        q = Gf.Matrix4d().SetLookAt(eye, target, up_axis).GetInverse().ExtractRotation().GetQuat()
        imag = q.GetImaginary()
        quat = np.array([float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])], dtype=float)
        quat = quat / max(1e-12, float(np.linalg.norm(quat)))
        return quat
    except Exception:
        # Fallback to a pure-numpy implementation.
        f = tgt - cam
        fn = float(np.linalg.norm(f))
        if fn < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        f = f / fn

        # USD camera looks along local -Z.
        z_axis = -f
        x_axis = np.cross(up, z_axis)
        xn = float(np.linalg.norm(x_axis))
        if xn < 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x_axis = x_axis / xn
        y_axis = np.cross(z_axis, x_axis)
        yn = float(np.linalg.norm(y_axis))
        if yn < 1e-9:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y_axis = y_axis / yn

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return _rotmat_to_quat_wxyz(R)


def _yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def _avoid_aabbs(
    pos: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    clearance_m: float,
    min_height_m: float | None,
) -> tuple[np.ndarray, bool]:
    """Push camera position out of any expanded AABB volumes."""
    if not aabbs:
        return pos, False
    cam = np.array(pos, dtype=float)
    adjusted = False
    margin = float(max(0.0, clearance_m))

    for aabb_min, aabb_max in aabbs:
        mn = np.asarray(aabb_min, dtype=float) - margin
        mx = np.asarray(aabb_max, dtype=float) + margin
        if (cam[0] >= mn[0] and cam[0] <= mx[0]) and (cam[1] >= mn[1] and cam[1] <= mx[1]) and (cam[2] >= mn[2] and cam[2] <= mx[2]):
            # Compute distances to each face (penetration depth to exit).
            dx_min = cam[0] - mn[0]
            dx_max = mx[0] - cam[0]
            dy_min = cam[1] - mn[1]
            dy_max = mx[1] - cam[1]
            dz_min = cam[2] - mn[2]
            dz_max = mx[2] - cam[2]
            # Move along the axis with smallest penetration to exit the box.
            options = [
                (dx_min, 0, mn[0]),
                (dx_max, 0, mx[0]),
                (dy_min, 1, mn[1]),
                (dy_max, 1, mx[1]),
                (dz_min, 2, mn[2]),
                (dz_max, 2, mx[2]),
            ]
            dist, axis, target = min(options, key=lambda t: t[0])
            if axis == 0:
                cam[0] = target
            elif axis == 1:
                cam[1] = target
            else:
                cam[2] = target
            adjusted = True

    if min_height_m is not None and cam[2] < float(min_height_m):
        cam[2] = float(min_height_m)
        adjusted = True

    return cam, adjusted


def _segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> bool:
    """Segment-AABB intersection (slab method)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    mn = np.asarray(mn, dtype=float)
    mx = np.asarray(mx, dtype=float)
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if p0[i] < mn[i] or p0[i] > mx[i]:
                return False
        else:
            inv = 1.0 / d[i]
            t1 = (mn[i] - p0[i]) * inv
            t2 = (mx[i] - p0[i]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmax < tmin:
                return False
    return True


def _ensure_line_of_sight(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    push_m: float,
    raise_m: float,
    max_iters: int,
    clearance_m: float,
    min_height_m: float | None,
) -> tuple[np.ndarray, bool]:
    """Adjust camera if line of sight to target is blocked by any AABB."""
    if not aabbs or max_iters <= 0:
        return cam_pos, False
    cam = np.array(cam_pos, dtype=float)
    adjusted = False
    for _ in range(int(max_iters)):
        blocked = False
        for mn, mx in aabbs:
            if _segment_intersects_aabb(cam, target_pos, mn, mx):
                blocked = True
                break
        if not blocked:
            break
        # Push camera outward and upward, then re-apply AABB avoidance.
        direction = cam - target_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
            norm = 1.0
        direction = direction / norm
        cam = cam + direction * float(push_m) + np.array([0.0, 0.0, float(raise_m)], dtype=float)
        cam, _ = _avoid_aabbs(cam, aabbs, clearance_m=clearance_m, min_height_m=min_height_m)
        adjusted = True
    return cam, adjusted


def _frame_quality_metrics(rgb_u8: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(rgb_u8, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    # Downsample for speed before computing edge energy.
    gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    gray = gray[::4, ::4]
    if gray.shape[0] >= 2 and gray.shape[1] >= 2:
        dv = np.abs(np.diff(gray, axis=0)).mean()
        dh = np.abs(np.diff(gray, axis=1)).mean()
        edge = float(0.5 * (dv + dh))
    else:
        edge = 0.0
    return {"mean": mean, "std": std, "edge": edge}


def _normalize_up_axis(axis: Any) -> str:
    ax = str(axis).strip().lower()
    if ax.startswith("y"):
        return "y"
    if ax.startswith("x"):
        return "x"
    return "z"


def _map_vec_our_to_stage(vec: np.ndarray, *, stage_up_axis: str) -> np.ndarray:
    """Map our Z-up XYZ vector into the stage axis convention (no unit scaling)."""
    v = np.asarray(vec, dtype=float).reshape(3)
    ax = _normalize_up_axis(stage_up_axis)
    if ax == "y":
        return np.array([v[0], v[2], v[1]], dtype=float)  # swap y<->z
    if ax == "x":
        return np.array([v[2], v[1], v[0]], dtype=float)  # swap x<->z
    return v


def _map_pos_m_to_stage_units(pos_m: np.ndarray, *, stage_up_axis: str, stage_meters_per_unit: float) -> np.ndarray:
    """Map our Z-up (meters) position into stage coordinates (stage units)."""
    mpu = float(stage_meters_per_unit) if stage_meters_per_unit else 1.0
    p_stage_m = _map_vec_our_to_stage(np.asarray(pos_m, dtype=float).reshape(3), stage_up_axis=stage_up_axis)
    return p_stage_m / max(1e-12, mpu)


def _map_vec_stage_to_our(vec: np.ndarray, *, stage_up_axis: str) -> np.ndarray:
    """Map a stage-axis vector into our internal Z-up XYZ convention (no unit scaling).

    NOTE: Our mapping is an axis swap, so it is its own inverse.
    """
    return _map_vec_our_to_stage(vec, stage_up_axis=stage_up_axis)


def _compute_stage_bbox(stage: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute an axis-aligned world-space bbox for the entire stage (stage units)."""
    try:
        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        aligned = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        return np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=float), np.array(
            [float(mx[0]), float(mx[1]), float(mx[2])], dtype=float
        )
    except Exception:
        return None


def _compute_prim_bbox(prim: Any, *, include_proxy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute an axis-aligned world-space bbox for a prim subtree (stage units)."""
    try:
        from pxr import Usd, UsdGeom

        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        if include_proxy:
            purposes.append(UsdGeom.Tokens.proxy)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=purposes, useExtentsHint=True)
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        return np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=float), np.array(
            [float(mx[0]), float(mx[1]), float(mx[2])], dtype=float
        )
    except Exception:
        return None


def _summarize_imageables_under(root_prim: Any, *, max_examples: int = 8) -> dict[str, Any]:
    """Summarize gprims under a prim: purposes, visibility, and a few example mesh paths."""
    summary: dict[str, Any] = {
        "gprims_total": 0,
        "purpose_counts": {"default": 0, "render": 0, "proxy": 0, "guide": 0, "other": 0},
        "invisible_gprims": 0,
        "mesh_examples": [],
    }
    try:
        from pxr import Usd, UsdGeom

        for prim in Usd.PrimRange(root_prim):
            if not prim or not prim.IsValid():
                continue
            try:
                img = UsdGeom.Imageable(prim)
            except Exception:
                continue
            try:
                if not img:
                    continue
            except Exception:
                pass

            # Heuristic: treat "gprims" as prims with a drawable type name.
            tn = str(prim.GetTypeName() or "")
            if tn not in {"Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"}:
                continue
            summary["gprims_total"] += 1

            try:
                purpose = str(img.GetPurposeAttr().Get() or "default")
            except Exception:
                purpose = "default"
            if purpose in summary["purpose_counts"]:
                summary["purpose_counts"][purpose] += 1
            else:
                summary["purpose_counts"]["other"] += 1

            try:
                vis = str(img.ComputeVisibility(Usd.TimeCode.Default()))
            except Exception:
                vis = "inherited"
            if vis == "invisible":
                summary["invisible_gprims"] += 1

            if tn == "Mesh" and len(summary["mesh_examples"]) < int(max_examples):
                summary["mesh_examples"].append({"path": str(prim.GetPath()), "purpose": purpose, "visibility": vis})
    except Exception:
        return summary
    return summary


def _force_purpose_and_visibility(root_prim: Any, *, purpose: str = "render") -> int:
    """Force a purpose/visibility override for all Imageable prims under root.

    Returns:
        Number of prims modified.
    """
    modified = 0
    try:
        from pxr import Usd, UsdGeom

        purpose_token = {
            "default": UsdGeom.Tokens.default_,
            "render": UsdGeom.Tokens.render,
            "proxy": UsdGeom.Tokens.proxy,
            "guide": UsdGeom.Tokens.guide,
        }.get(str(purpose).strip().lower(), UsdGeom.Tokens.render)

        for prim in Usd.PrimRange(root_prim):
            if not prim or not prim.IsValid():
                continue
            try:
                img = UsdGeom.Imageable(prim)
            except Exception:
                continue
            try:
                if not img:
                    continue
            except Exception:
                pass
            try:
                img.GetPurposeAttr().Set(purpose_token)
                img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                modified += 1
            except Exception:
                continue
    except Exception:
        return 0
    return modified


def _infer_anchor_from_stage_bbox(
    bbox_min_stage_units: np.ndarray,
    bbox_max_stage_units: np.ndarray,
    *,
    stage_up_axis: str,
    stage_meters_per_unit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer an anchor position (our meters) from a stage bbox.

    Returns:
        anchor_m_our: XY at bbox center, Z at bbox floor (stage up axis).
        center_m_our: full bbox center in our meters (Z-up).
        extent_m_our: bbox size in our meters (Z-up).
    """
    mpu = float(stage_meters_per_unit) if stage_meters_per_unit else 1.0
    mn_m_stage = np.asarray(bbox_min_stage_units, dtype=float).reshape(3) * mpu
    mx_m_stage = np.asarray(bbox_max_stage_units, dtype=float).reshape(3) * mpu
    center_m_stage = 0.5 * (mn_m_stage + mx_m_stage)
    extent_m_stage = mx_m_stage - mn_m_stage

    center_m_our = _map_vec_stage_to_our(center_m_stage, stage_up_axis=stage_up_axis)
    extent_m_our = np.abs(_map_vec_stage_to_our(extent_m_stage, stage_up_axis=stage_up_axis))

    ax = _normalize_up_axis(stage_up_axis)
    up_idx = {"x": 0, "y": 1, "z": 2}[ax]
    floor_m_our_z = float(mn_m_stage[up_idx])
    anchor_m_our = np.array([float(center_m_our[0]), float(center_m_our[1]), floor_m_our_z], dtype=float)
    return anchor_m_our, center_m_our, extent_m_our


def _recommend_cam_offset_from_extent(
    extent_m_our: np.ndarray,
    *,
    distance_scale: float = 0.85,
    height_scale: float = 0.45,
    min_distance_m: float = 8.0,
    min_height_m: float = 2.0,
) -> np.ndarray:
    """Heuristic third-person offset that captures an overview of the stage."""
    ext = np.asarray(extent_m_our, dtype=float).reshape(3)
    horiz = float(max(ext[0], ext[1], 1e-6))
    dist = max(float(min_distance_m), float(distance_scale) * horiz)
    height = max(float(min_height_m), float(height_scale) * horiz)
    return np.array([-dist, -dist, height], dtype=float)


def _yaw_quat_wxyz_about_up_axis(yaw_rad: float, *, stage_up_axis: str) -> np.ndarray:
    """Yaw rotation quaternion about the stage up axis (wxyz)."""
    half = 0.5 * float(yaw_rad)
    c = math.cos(half)
    s = math.sin(half)
    ax = _normalize_up_axis(stage_up_axis)
    if ax == "y":
        return np.array([c, 0.0, s, 0.0], dtype=float)
    if ax == "x":
        return np.array([c, s, 0.0, 0.0], dtype=float)
    return np.array([c, 0.0, 0.0, s], dtype=float)


def _is_url(path: str) -> bool:
    try:
        scheme = urlparse(path).scheme
    except Exception:
        return False
    return scheme in {"http", "https", "omniverse", "file"}

def _run_cmd_lines(cmd: list[str], *, timeout_s: float = 2.0) -> list[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=float(timeout_s))
        text = out.decode("utf-8", errors="replace")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    except Exception:
        return []


def _auto_select_single_gpu(*, prefer_high_index: bool = True) -> int | None:
    """Pick a single mostly-idle GPU (prefer higher index) using nvidia-smi.

    Returns:
        GPU index, or None if nvidia-smi unavailable / parse failed.
    """
    q = _run_cmd_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=3.0,
    )
    if not q:
        return None

    # If compute apps are running, avoid those GPUs when possible.
    uuid_lines = _run_cmd_lines(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        timeout_s=3.0,
    )
    uuid_to_index: dict[str, int] = {}
    for ln in uuid_lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2:
            try:
                uuid_to_index[parts[1]] = int(parts[0])
            except Exception:
                continue

    apps = _run_cmd_lines(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        timeout_s=3.0,
    )
    busy_indices: set[int] = set()
    for ln in apps:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 1:
            idx = uuid_to_index.get(parts[0])
            if idx is not None:
                busy_indices.add(int(idx))

    candidates: list[tuple[int, int, int]] = []
    for ln in q:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            util = int(parts[1])
            mem_mb = int(parts[2])
        except Exception:
            continue
        # "Idle-enough" heuristic (render start-up will spike anyway).
        if util <= 5 and mem_mb <= 1024:
            candidates.append((idx, util, mem_mb))

    if not candidates:
        return None
    # Prefer higher or lower indices depending on config.
    candidates.sort(key=lambda t: t[0], reverse=bool(prefer_high_index))
    non_busy = [c for c in candidates if c[0] not in busy_indices]
    pick = non_busy[0] if non_busy else candidates[0]
    return int(pick[0])


def _get_valid_rgba(camera: Any, tick: Any, *, max_tries: int = 30) -> np.ndarray:
    """Try to obtain a valid HxWxC rgba buffer from the camera.

    Depending on the Isaac experience and startup sequence, `Camera.get_rgba()` can return
    an empty array for some frames. We run `tick()` to let the pipeline warm up.
    """
    last = None
    for _ in range(max(1, int(max_tries))):
        rgba = camera.get_rgba()
        last = rgba
        arr = np.asarray(rgba) if rgba is not None else None
        if arr is not None and arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0 and arr.shape[2] >= 3:
            return rgba
        tick()
    details = {"type": type(last).__name__}
    try:
        last_arr = np.asarray(last)
        details.update({"shape": tuple(int(x) for x in last_arr.shape), "dtype": str(last_arr.dtype), "size": int(last_arr.size)})
    except Exception:
        pass
    raise RuntimeError(f"Failed to capture a valid rgba buffer after {max_tries} tries (last={details}).")


def _normalize_rep_rgb_to_u8(arr: Any) -> np.ndarray:
    """Normalize replicator annotator `rgb` output to uint8 RGB (HxWx3)."""
    img = np.asarray(arr)
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC from replicator rgb annotator, got shape={img.shape}")
    if img.shape[2] >= 3:
        img = img[:, :, :3]
    # Replicator 'rgb' typically returns uint8 already; handle float just in case.
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0.0, 255.0) if img.max() > 1.5 else np.clip(img, 0.0, 1.0) * 255.0
    return img.astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac showcase runner: produce high-quality rendered frames.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output", default="runs_isaac", help="Output root folder under Isaac/")
    parser.add_argument("--name", default="showcase", help="Run name.")
    parser.add_argument(
        "--experience",
        default="",
        help="Optional Kit experience (.kit). If empty, use Isaac Sim default (recommended for stable viewport capture).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    output_root = (Path(__file__).resolve().parents[1] / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run = prepare_run(output_root=output_root, name=args.name, extra_meta={"config_path": str(cfg_path), "mode": "showcase"})
    write_json(run.run_dir / "config.json", cfg)
    progress_log = run.run_dir / "progress.log"
    error_log = run.run_dir / "error.txt"

    def _progress(msg: str) -> None:
        line = msg.strip()
        if not line:
            return
        print(line, flush=True)
        try:
            with progress_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    sim_cfg = cfg.get("sim", {})
    scene_cfg = cfg.get("scene", {})
    uav_cfg = cfg.get("uav", {})
    capture_cfg = cfg.get("capture", {})

    headless = bool(sim_cfg.get("headless", True))
    physics_dt_s = float(sim_cfg.get("physics_dt_s", 1.0 / 60.0))
    rendering_dt_s = float(sim_cfg.get("rendering_dt_s", physics_dt_s))
    steps = int(sim_cfg.get("steps", 240))
    fail_on_bad_frames = bool(sim_cfg.get("fail_on_bad_frames", False))

    # Resource selection (single-GPU, prefer higher index).
    requested_gpu = sim_cfg.get("gpu", sim_cfg.get("active_gpu", None))
    auto_select_gpu = bool(sim_cfg.get("auto_select_gpu", True))
    prefer_high_gpu = bool(sim_cfg.get("prefer_high_gpu", True))
    if requested_gpu is None and auto_select_gpu:
        requested_gpu = _auto_select_single_gpu(prefer_high_index=prefer_high_gpu)
    if requested_gpu is not None:
        try:
            requested_gpu = int(requested_gpu)
        except Exception:
            requested_gpu = None
    limit_cpu_threads = int(sim_cfg.get("limit_cpu_threads", 16))

    seed = int(sim_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    ground_extent_m = float(scene_cfg.get("ground_extent_m", 200.0))
    building_count = int(scene_cfg.get("building_count", 25))
    building_area_m = float(scene_cfg.get("building_area_m", 80.0))
    clear_radius_m = scene_cfg.get("clear_radius_m", None)
    camera_clearance_m = float(scene_cfg.get("camera_clearance_m", 6.0))
    building_height_min = float(scene_cfg.get("building_height_min_m", 6.0))
    building_height_max = float(scene_cfg.get("building_height_max_m", 26.0))
    ground_color = np.array(scene_cfg.get("ground_color", [0.18, 0.19, 0.20]), dtype=float)
    building_color = np.array(scene_cfg.get("building_color", [0.45, 0.47, 0.50]), dtype=float)
    building_palette = scene_cfg.get("building_palette", None)
    if building_palette:
        try:
            building_palette = [np.array(c, dtype=float) for c in building_palette]
        except Exception:
            building_palette = None

    altitude_m = float(uav_cfg.get("altitude_m", 8.0))
    radius_m = float(uav_cfg.get("radius_m", 25.0))
    period_s = float(uav_cfg.get("period_s", 6.0))
    uav_scale = float(uav_cfg.get("scale", 1.0))
    uav_force_render_purpose = bool(uav_cfg.get("force_render_purpose", False))
    uav_usd = str(uav_cfg.get("usd", uav_cfg.get("asset_usd", uav_cfg.get("usd_url", "")))).strip()
    uav_prim_path = str(uav_cfg.get("prim_path", "/World/UAV")).strip() or "/World/UAV"
    uav_center_on_stage = bool(uav_cfg.get("center_on_stage_bbox", True))

    cap_enabled = bool(capture_cfg.get("enabled", True))
    cap_res = capture_cfg.get("resolution", [1920, 1080])
    cap_res = (int(cap_res[0]), int(cap_res[1]))
    cap_backend = str(capture_cfg.get("backend", "replicator")).strip().lower()
    cap_every_n = int(capture_cfg.get("every_n_steps", 1))
    cap_every_n = max(1, cap_every_n)
    warmup_steps = int(capture_cfg.get("warmup_steps", 10))
    screenshot_step = int(capture_cfg.get("screenshot_step", max(0, steps // 2)))
    screenshot_step = int(np.clip(screenshot_step, 0, max(0, steps - 1)))

    cam_cfg = capture_cfg.get("camera", {})
    cam_offset = np.array(cam_cfg.get("offset_m", [-35.0, -35.0, 18.0]), dtype=float)
    cam_up = np.array(cam_cfg.get("up", [0.0, 0.0, 1.0]), dtype=float)
    cam_focal_mm = float(cam_cfg.get("focal_length_mm", 24.0))
    cam_target_offset = np.array(cam_cfg.get("target_offset_m", [0.0, 0.0, 0.0]), dtype=float)
    cam_follow_uav = bool(cam_cfg.get("follow_uav", True))
    cam_auto_from_bbox = bool(cam_cfg.get("auto_from_stage_bbox", False))
    cam_auto_distance_scale = float(cam_cfg.get("auto_distance_scale", 0.85))
    cam_auto_height_scale = float(cam_cfg.get("auto_height_scale", 0.45))
    cam_auto_min_distance_m = float(cam_cfg.get("auto_min_distance_m", 8.0))
    cam_auto_min_height_m = float(cam_cfg.get("auto_min_height_m", 2.0))
    cam_min_height_m = cam_cfg.get("min_height_m", None)
    if cam_min_height_m is None:
        cam_min_height_m = float(altitude_m + 8.0)
    los_check = bool(cam_cfg.get("los_check", True))
    los_push_m = float(cam_cfg.get("los_push_m", 12.0))
    los_raise_m = float(cam_cfg.get("los_raise_m", 6.0))
    los_max_iters = int(cam_cfg.get("los_max_iters", 3))

    quality_cfg = capture_cfg.get("quality", {})
    min_mean = float(quality_cfg.get("min_mean", 5.0))
    max_mean = float(quality_cfg.get("max_mean", 245.0))
    min_std = float(quality_cfg.get("min_std", 5.0))
    min_edge = float(quality_cfg.get("min_edge", 1.0))

    render_cfg = cfg.get("render", {})
    dome_intensity = float(render_cfg.get("dome_intensity", 1200.0))
    sun_intensity = float(render_cfg.get("sun_intensity", 8000.0))
    sun_color = np.array(render_cfg.get("sun_color", [1.0, 0.98, 0.95]), dtype=float)
    sun_dir = np.array(render_cfg.get("sun_direction", [0.3, 0.5, -1.0]), dtype=float)  # points from light to scene
    use_hdri = bool(render_cfg.get("use_hdri", True))
    dome_texture = str(render_cfg.get("dome_texture", "")).strip() if use_hdri else ""
    if use_hdri and not dome_texture:
        # Prefer a high-quality HDRI that ships inside Isaac Sim install (no redistribution).
        isaac_root = os.environ.get("ISAAC_PATH") or os.environ.get("ISAACSIM_ROOT") or ""
        if isaac_root:
            extscache = Path(isaac_root) / "extscache"
            if extscache.exists():
                for name in ("photo_studio_01_4k.hdr", "StinsonBeach.hdr", "sunflowers.hdr"):
                    try:
                        hit = next(extscache.rglob(name))
                        dome_texture = str(hit)
                        break
                    except StopIteration:
                        continue

    from isaacsim import SimulationApp

    experience = args.experience.strip()
    launcher_cfg: Dict[str, Any] = {"headless": headless, "multi_gpu": False, "limit_cpu_threads": int(limit_cpu_threads)}
    if requested_gpu is not None:
        launcher_cfg["active_gpu"] = int(requested_gpu)
        launcher_cfg["physics_gpu"] = int(requested_gpu)
        _progress(f"[info] Using single GPU index: {int(requested_gpu)} (prefer_high_gpu={prefer_high_gpu})")
    simulation_app = SimulationApp(launcher_cfg, experience=experience)
    try:
        import omni

        from isaacsim.core.api.objects import VisualCuboid
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import get_current_stage, open_stage, set_stage_up_axis, set_stage_units
        from isaacsim.sensors.camera import Camera
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

        # Enable camera extensions in headless mode (best-effort).
        try:
            enable_extension("omni.sensors.nv.camera")
        except Exception:
            pass

        simulation_app.update()
        omni.usd.get_context().new_stage()
        simulation_app.update()

        # Optional: open an official or user-provided USD stage (no asset redistribution).
        stage_usd = str(scene_cfg.get("stage_usd", "")).strip()
        if stage_usd:
            stage_path = stage_usd
            if stage_usd.startswith("/Isaac/"):
                try:
                    from isaacsim.storage.native import get_assets_root_path

                    assets_root = get_assets_root_path()
                except Exception:
                    assets_root = None
                if not assets_root:
                    raise RuntimeError(
                        "scene.stage_usd points to an Isaac asset path, but assets root was not found. "
                        "Configure Omniverse/Nucleus assets or use a local USD path."
                    )
                stage_path = str(assets_root) + stage_usd
            else:
                if _is_url(stage_usd):
                    stage_path = stage_usd
                else:
                    stage_file = Path(stage_path).expanduser()
                    if not stage_file.is_absolute():
                        stage_file = cfg_dir / stage_file
                    stage_path = str(stage_file.resolve())
            _progress(f"[info] Opening stage: {stage_path}")
            if not open_stage(stage_path):
                raise RuntimeError(f"Failed to open stage: {stage_path}")
            simulation_app.update()

        stage = get_current_stage()
        # Stage conventions: if we're building a procedural scene, force Z-up + meters for stability.
        # If we opened an external stage, keep its authored conventions and adapt our placements.
        if not stage_usd:
            try:
                set_stage_up_axis("z")
            except Exception:
                pass
            try:
                set_stage_units(1.0)
            except Exception:
                pass
            stage_up_axis = "z"
            stage_meters_per_unit = 1.0
        else:
            try:
                stage_up_axis = _normalize_up_axis(UsdGeom.GetStageUpAxis(stage))
            except Exception:
                stage_up_axis = "z"
            try:
                stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
            except Exception:
                stage_meters_per_unit = 1.0
            if stage_meters_per_unit <= 0.0:
                stage_meters_per_unit = 1.0

        # External stages (and HTTPS USDs) may be read-only on disk; always author edits into the session layer.
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
        except Exception:
            pass

        # Stage bbox-based anchor (helps keep UAV/camera inside the visible environment for authored stages).
        stage_bbox_min_stage_units = None
        stage_bbox_max_stage_units = None
        stage_center_m = np.zeros(3, dtype=float)
        stage_extent_m = np.zeros(3, dtype=float)
        scene_anchor_m = np.zeros(3, dtype=float)
        if stage_usd and uav_center_on_stage:
            bbox = _compute_stage_bbox(stage)
            if bbox is not None:
                stage_bbox_min_stage_units, stage_bbox_max_stage_units = bbox
                scene_anchor_m, stage_center_m, stage_extent_m = _infer_anchor_from_stage_bbox(
                    stage_bbox_min_stage_units,
                    stage_bbox_max_stage_units,
                    stage_up_axis=stage_up_axis,
                    stage_meters_per_unit=stage_meters_per_unit,
                )
                _progress(
                    "[info] Stage bbox (m):"
                    f" center={np.round(stage_center_m, 3).tolist()} extent={np.round(stage_extent_m, 3).tolist()}"
                )
                if cam_auto_from_bbox:
                    cam_offset = _recommend_cam_offset_from_extent(
                        stage_extent_m,
                        distance_scale=cam_auto_distance_scale,
                        height_scale=cam_auto_height_scale,
                        min_distance_m=cam_auto_min_distance_m,
                        min_height_m=cam_auto_min_height_m,
                    )
                    _progress(f"[info] Camera offset auto-from-bbox: {np.round(cam_offset, 3).tolist()} (meters)")
        else:
            scene_anchor_m = np.zeros(3, dtype=float)

        def _stage_has_lights() -> bool:
            try:
                for prim in stage.Traverse():
                    tn = prim.GetTypeName()
                    if tn in {
                        "DomeLight",
                        "DistantLight",
                        "RectLight",
                        "SphereLight",
                        "DiskLight",
                        "CylinderLight",
                    }:
                        return True
            except Exception:
                pass
            return False

        # Lighting: for procedural scenes we always add lights; for external stages we add lights only when missing.
        stage_has_lights = _stage_has_lights()
        add_lights_if_missing = bool(render_cfg.get("add_lighting_if_missing", True))
        if (not stage_has_lights) and (not stage_usd or add_lights_if_missing):
            dome_path = "/World/DomeLight" if not stage_usd else "/World/ShowcaseDomeLight"
            sun_xform_path = "/World/SunXform" if not stage_usd else "/World/ShowcaseSunXform"
            sun_path = f"{sun_xform_path}/Sun"

            dome = UsdLux.DomeLight.Define(stage, dome_path)
            dome.CreateIntensityAttr(dome_intensity)
            dome.CreateColorAttr(Gf.Vec3f(float(sun_color[0]), float(sun_color[1]), float(sun_color[2])))
            if dome_texture:
                try:
                    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(str(dome_texture)))
                    dome.CreateTextureFormatAttr("latlong")
                    _progress(f"[info] DomeLight HDRI: {dome_texture}")
                except Exception as exc:
                    _progress(f"[warn] Failed to set DomeLight HDRI ({type(exc).__name__}: {exc})")

            # A simple distant "sun" light.
            sun_xform = UsdGeom.Xform.Define(stage, sun_xform_path)
            sun = UsdLux.DistantLight.Define(stage, sun_path)
            sun.CreateIntensityAttr(sun_intensity)
            sun.CreateColorAttr(Gf.Vec3f(float(sun_color[0]), float(sun_color[1]), float(sun_color[2])))
            # Orient the sun so its -Z roughly points towards the scene.
            d = np.asarray(sun_dir, dtype=float).reshape(3)
            dn = float(np.linalg.norm(d))
            if dn > 1e-6:
                d = d / dn
            # Build a rotation that maps local -Z to direction d.
            z_axis = -d
            x_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), z_axis)
            if float(np.linalg.norm(x_axis)) < 1e-6:
                x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                x_axis = x_axis / float(np.linalg.norm(x_axis))
            y_axis = np.cross(z_axis, x_axis)
            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            q_sun = _rotmat_to_quat_wxyz(R)
            sun_xformable = UsdGeom.Xformable(sun_xform.GetPrim())
            sun_xformable.AddOrientOp().Set(
                Gf.Quatf(float(q_sun[0]), Gf.Vec3f(float(q_sun[1]), float(q_sun[2]), float(q_sun[3])))
            )

        # Procedural environment is only created when we don't open an external stage USD.
        building_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        if not stage_usd:
            # Procedural fallback: flat ground + simple buildings.
            VisualCuboid(
                prim_path="/World/ground",
                name="ground",
                position=np.array([0.0, 0.0, -0.05], dtype=float),
                scale=np.array([ground_extent_m, ground_extent_m, 0.1], dtype=float),
                size=1.0,
                color=ground_color,
            )

            # Ground grid (adds edges/structure so the overview shot doesn't look flat).
            grid_cfg = scene_cfg.get("ground_grid", {}) or {}
            grid_enabled = bool(grid_cfg.get("enabled", True))
            grid_extent_m = float(grid_cfg.get("extent_m", min(0.45 * float(ground_extent_m), 200.0)))
            grid_spacing_m = float(grid_cfg.get("spacing_m", 20.0))
            grid_line_width_m = float(grid_cfg.get("line_width_m", 0.5))
            grid_line_height_m = float(grid_cfg.get("line_height_m", 0.06))
            grid_color = np.array(grid_cfg.get("color", [0.85, 0.85, 0.85]), dtype=float)
            if grid_enabled and grid_extent_m > 0.0 and grid_spacing_m > 0.0:
                z_center = 0.02 + 0.5 * grid_line_height_m
                n_lines = int(max(2, round((2.0 * grid_extent_m) / grid_spacing_m) + 1))
                coords = np.linspace(-grid_extent_m, grid_extent_m, n_lines)
                for i, x in enumerate(coords):
                    VisualCuboid(
                        prim_path=f"/World/grid_x_{i:03d}",
                        name=f"grid_x_{i:03d}",
                        position=np.array([float(x), 0.0, z_center], dtype=float),
                        scale=np.array([grid_line_width_m, 2.0 * grid_extent_m, grid_line_height_m], dtype=float),
                        size=1.0,
                        color=grid_color,
                    )
                for i, y in enumerate(coords):
                    VisualCuboid(
                        prim_path=f"/World/grid_y_{i:03d}",
                        name=f"grid_y_{i:03d}",
                        position=np.array([0.0, float(y), z_center], dtype=float),
                        scale=np.array([2.0 * grid_extent_m, grid_line_width_m, grid_line_height_m], dtype=float),
                        size=1.0,
                        color=grid_color,
                    )

            # Buildings.
            min_clear = float(radius_m) + 20.0
            if clear_radius_m is None:
                clear_radius_m = min_clear
            clear_radius_m = float(max(min_clear, clear_radius_m, 0.0))
            building_r_min = float(clear_radius_m)
            building_r_max = float(max(building_r_min + 40.0, building_area_m))

            # Bias buildings into the "in front of camera" direction to create a skyline background.
            bias_prob = float(scene_cfg.get("building_bias_prob", 0.75))
            bias_arc_deg = float(scene_cfg.get("building_bias_arc_deg", 110.0))
            bias_arc = math.radians(max(1.0, min(180.0, bias_arc_deg)))
            view_dir_xy = np.array([-float(cam_offset[0]), -float(cam_offset[1])], dtype=float)
            view_norm = float(np.linalg.norm(view_dir_xy))
            view_ang = float(math.atan2(view_dir_xy[1], view_dir_xy[0])) if view_norm > 1e-6 else 0.0
            for i in range(building_count):
                x = 0.0
                y = 0.0
                if float(rng.random()) < bias_prob:
                    ang = float(view_ang + rng.uniform(-bias_arc, bias_arc))
                else:
                    ang = float(rng.uniform(0.0, 2.0 * math.pi))
                rr = float(rng.uniform(building_r_min, building_r_max))
                x = rr * math.cos(ang)
                y = rr * math.sin(ang)
                sx = float(rng.uniform(4.0, 14.0))
                sy = float(rng.uniform(4.0, 14.0))
                h = float(rng.uniform(building_height_min, building_height_max))
                color = building_color
                if building_palette:
                    color = building_palette[int(rng.integers(0, len(building_palette)))]
                hx, hy = 0.5 * sx, 0.5 * sy
                building_aabbs.append(
                    (
                        np.array([x - hx, y - hy, 0.0], dtype=float),
                        np.array([x + hx, y + hy, h], dtype=float),
                    )
                )
                VisualCuboid(
                    prim_path=f"/World/building_{i:03d}",
                    name=f"building_{i:03d}",
                    position=np.array([x, y, h / 2.0], dtype=float),
                    scale=np.array([sx, sy, h], dtype=float),
                    size=1.0,
                    color=color,
                )

        # UAV: prefer an official USD asset for high-quality visuals; fall back to a procedural silhouette.
        uav_pos0_m = scene_anchor_m + np.array([0.0, 0.0, float(altitude_m)], dtype=float)
        uav_pos0 = _map_pos_m_to_stage_units(
            uav_pos0_m, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
        )

        uav_mode = "procedural"
        uav_asset_path = None
        uav_xformable = None
        uav_translate_op = None
        uav_orient_op = None
        body = None
        arm_x = None
        arm_y = None
        motors = []
        motor_offsets = []

        def _resolve_any_path(raw: str) -> str:
            raw = str(raw).strip()
            if not raw:
                return raw
            if raw.startswith("/Isaac/"):
                try:
                    from isaacsim.storage.native import get_assets_root_path

                    assets_root = get_assets_root_path()
                except Exception:
                    assets_root = None
                if not assets_root:
                    raise RuntimeError(
                        "UAV USD is a /Isaac/... path but assets root was not found. "
                        "Use an official HTTPS URL or configure a local assets root/Nucleus."
                    )
                return str(assets_root) + raw
            if _is_url(raw):
                return raw
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = cfg_dir / p
            return str(p.resolve())

        if uav_usd:
            try:
                uav_mode = "usd_asset"
                uav_asset_path = _resolve_any_path(uav_usd)
                _progress(f"[info] UAV USD asset: {uav_asset_path} -> prim={uav_prim_path}")

                try:
                    UsdGeom.Xform.Define(stage, uav_prim_path)
                except Exception:
                    pass
                uav_prim = stage.GetPrimAtPath(uav_prim_path)
                if not uav_prim or not uav_prim.IsValid():
                    raise RuntimeError(f"Failed to create/find UAV prim: {uav_prim_path}")
                try:
                    uav_prim.GetReferences().ClearReferences()
                except Exception:
                    pass
                # NOTE: UsdReferences.AddReference expects a string asset path (passing Sdf.AssetPath may throw).
                uav_prim.GetReferences().AddReference(str(uav_asset_path))
                uav_xformable = UsdGeom.Xformable(uav_prim)

                translate_ops = [
                    op for op in uav_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ]
                uav_translate_op = translate_ops[0] if translate_ops else uav_xformable.AddTranslateOp()
                orient_ops = [op for op in uav_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
                uav_orient_op = orient_ops[0] if orient_ops else uav_xformable.AddOrientOp()
                scale_ops = [op for op in uav_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                scale_op = scale_ops[0] if scale_ops else uav_xformable.AddScaleOp()
                scale_op.Set(Gf.Vec3f(float(uav_scale), float(uav_scale), float(uav_scale)))

                # IMPORTANT: enforce an op order that will NOT scale the translation (otherwise the UAV can end up kilometers away).
                # USD xform ops are applied right-to-left; keeping translate first ensures it is applied last and does not get scaled.
                try:
                    uav_xformable.SetXformOpOrder([uav_translate_op, uav_orient_op, scale_op])
                    _progress("[info] UAV xformOpOrder set to: translate -> orient -> scale")
                except Exception as exc:
                    _progress(f"[warn] Failed to set UAV xformOpOrder ({type(exc).__name__}: {exc})")

                uav_translate_op.Set(Gf.Vec3d(float(uav_pos0[0]), float(uav_pos0[1]), float(uav_pos0[2])))
                uav_orient_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

                # Best-effort: ensure referenced payloads are loaded (headless runs sometimes need this).
                try:
                    uav_prim.Load()
                except Exception:
                    pass
                simulation_app.update()

                if uav_force_render_purpose:
                    try:
                        changed = _force_purpose_and_visibility(uav_prim, purpose="render")
                        _progress(f"[info] Forced UAV purpose/visibility overrides (count={changed})")
                    except Exception as exc:
                        _progress(f"[warn] Failed to force UAV purpose/visibility ({type(exc).__name__}: {exc})")

                # Log UAV bbox/visibility hints (critical for outdoor showcase where the UAV may be microscopic).
                try:
                    uav_bbox = _compute_prim_bbox(uav_prim, include_proxy=True)
                    uav_bbox_render = _compute_prim_bbox(uav_prim, include_proxy=False)
                    if uav_bbox is None:
                        _progress(
                            "[warn] UAV bbox: <failed to compute> (may indicate unloaded payloads or no renderable geometry)"
                        )
                    else:
                        uav_bbox_min, uav_bbox_max = uav_bbox
                        uav_center_m_stage = 0.5 * (uav_bbox_min + uav_bbox_max) * float(stage_meters_per_unit)
                        uav_extent_m_stage = (uav_bbox_max - uav_bbox_min) * float(stage_meters_per_unit)
                        uav_center_m = _map_vec_stage_to_our(uav_center_m_stage, stage_up_axis=stage_up_axis)
                        uav_extent_m = np.abs(_map_vec_stage_to_our(uav_extent_m_stage, stage_up_axis=stage_up_axis))
                        _progress(
                            "[info] UAV bbox (m):"
                            f" center={np.round(uav_center_m, 4).tolist()} extent={np.round(uav_extent_m, 4).tolist()}"
                        )
                        if float(np.max(uav_extent_m)) < 0.05:
                            _progress(
                                "[warn] UAV appears extremely small in meters; consider increasing `uav.scale` (e.g., 50–200) "
                                "or reducing camera distance."
                            )
                    if uav_bbox_render is None:
                        _progress("[warn] UAV bbox (render-only): <none> (may indicate geometry is authored as proxy/guide)")
                    else:
                        mn, mx = uav_bbox_render
                        ext = (mx - mn) * float(stage_meters_per_unit)
                        ext_our = np.abs(_map_vec_stage_to_our(ext, stage_up_axis=stage_up_axis))
                        _progress(f"[info] UAV bbox (render-only, m) extent={np.round(ext_our, 4).tolist()}")
                except Exception as exc:
                    _progress(f"[warn] UAV bbox debug failed ({type(exc).__name__}: {exc})")

                try:
                    _progress(f"[info] UAV resetXformStack={bool(uav_xformable.GetResetXformStack())}")
                except Exception:
                    pass

                try:
                    child_summaries = []
                    for idx, child in enumerate(uav_prim.GetChildren()):
                        child_summaries.append(f"{child.GetName()}:{child.GetTypeName()}")
                        if idx >= 7:
                            break
                    if child_summaries:
                        _progress(f"[info] UAV children (first {len(child_summaries)}): " + ", ".join(child_summaries))
                except Exception:
                    pass

                try:
                    img_summary = _summarize_imageables_under(uav_prim, max_examples=6)
                    _progress(
                        "[info] UAV gprims summary: "
                        f"total={img_summary.get('gprims_total')}, "
                        f"purposes={img_summary.get('purpose_counts')}, "
                        f"invisible={img_summary.get('invisible_gprims')}"
                    )
                    examples = img_summary.get("mesh_examples") or []
                    for ex in examples:
                        _progress(f"[info] UAV mesh example: {ex}")
                except Exception:
                    pass
            except Exception as exc:
                _progress(f"[warn] Failed to reference UAV USD; falling back to procedural UAV ({type(exc).__name__}: {exc})")
                try:
                    stage.RemovePrim(uav_prim_path)
                except Exception:
                    pass
                uav_usd = ""
                uav_mode = "procedural"
                uav_asset_path = None
                uav_xformable = None
                uav_translate_op = None
                uav_orient_op = None
        else:
            # Procedural silhouette (kept as a fallback for environments without external assets).
            uav_paths = {
                "body": "/World/UAV/body",
                "arm_x": "/World/UAV/arm_x",
                "arm_y": "/World/UAV/arm_y",
                "m1": "/World/UAV/m1",
                "m2": "/World/UAV/m2",
                "m3": "/World/UAV/m3",
                "m4": "/World/UAV/m4",
            }
            body = VisualCuboid(
                prim_path=uav_paths["body"],
                name="uav_body",
                position=uav_pos0,
                scale=np.array([0.9, 0.9, 0.2], dtype=float) * uav_scale,
                size=1.0,
                color=np.array([0.08, 0.08, 0.09], dtype=float),
            )
            arm_x = VisualCuboid(
                prim_path=uav_paths["arm_x"],
                name="uav_arm_x",
                position=uav_pos0,
                scale=np.array([2.2, 0.18, 0.12], dtype=float) * uav_scale,
                size=1.0,
                color=np.array([0.15, 0.15, 0.16], dtype=float),
            )
            arm_y = VisualCuboid(
                prim_path=uav_paths["arm_y"],
                name="uav_arm_y",
                position=uav_pos0,
                scale=np.array([0.18, 2.2, 0.12], dtype=float) * uav_scale,
                size=1.0,
                color=np.array([0.15, 0.15, 0.16], dtype=float),
            )
            motor_offsets = [
                np.array([1.1, 0.0, 0.06], dtype=float) * uav_scale,
                np.array([0.0, 1.1, 0.06], dtype=float) * uav_scale,
                np.array([-1.1, 0.0, 0.06], dtype=float) * uav_scale,
                np.array([0.0, -1.1, 0.06], dtype=float) * uav_scale,
            ]
            motor_colors = [
                np.array([0.90, 0.20, 0.20], dtype=float),
                np.array([0.20, 0.90, 0.20], dtype=float),
                np.array([0.20, 0.20, 0.90], dtype=float),
                np.array([0.90, 0.90, 0.20], dtype=float),
            ]
            for idx, (off, col) in enumerate(zip(motor_offsets, motor_colors, strict=True), start=1):
                pos_stage = _map_pos_m_to_stage_units(
                    uav_pos0_m + off, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
                )
                motors.append(
                    VisualCuboid(
                        prim_path=uav_paths[f"m{idx}"],
                        name=f"uav_m{idx}",
                        position=pos_stage,
                        scale=np.array([0.25, 0.25, 0.10], dtype=float) * uav_scale,
                        size=1.0,
                        color=col,
                    )
                )

        # Camera (single cinematic view).
        artifacts_dir = run.run_dir / "artifacts"
        frames_dir = artifacts_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        _progress(f"[info] Artifacts dir: {artifacts_dir}")

        cam_prim_path = "/World/ShowcaseCamera"
        cam_up_stage = _map_vec_our_to_stage(cam_up, stage_up_axis=stage_up_axis)
        cam_pos0_m = uav_pos0_m + cam_offset
        cam_target0_m = uav_pos0_m + cam_target_offset
        cam_pos0 = _map_pos_m_to_stage_units(
            cam_pos0_m, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
        )
        cam_target0 = _map_pos_m_to_stage_units(
            cam_target0_m, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
        )
        cam_q0 = _look_at_quat_wxyz(cam_pos0, cam_target0, cam_up_stage)

        # Ensure the USD camera prim exists before we create the render product.
        try:
            UsdGeom.Camera.Define(stage, cam_prim_path)
        except Exception:
            pass
        # Initialize camera transform explicitly (important when using Replicator capture).
        try:
            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            xformable = UsdGeom.Xformable(cam_prim)
            translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            translate_op = translate_ops[0] if translate_ops else xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(cam_pos0[0]), float(cam_pos0[1]), float(cam_pos0[2])))
            orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
            orient_op = orient_ops[0] if orient_ops else xformable.AddOrientOp()
            orient_op.Set(Gf.Quatf(float(cam_q0[0]), Gf.Vec3f(float(cam_q0[1]), float(cam_q0[2]), float(cam_q0[3]))))
        except Exception:
            pass

        # Capture backend: prefer Replicator annotator stepping (more reliable in headless).
        render_product_path = None
        render_product_source = None
        viewport_api = None
        rep = None
        rep_rgb_annot = None
        rep_rp = None
        camera = None
        if cap_backend in {"rep", "replicator"}:
            try:
                enable_extension("omni.replicator.core")
                import omni.replicator.core as rep  # type: ignore[no-redef]

                rep_rp = rep.create.render_product(cam_prim_path, resolution=cap_res, name="Showcase")
                render_product_path = getattr(rep_rp, "path", None) or str(rep_rp)
                render_product_source = "replicator"
                rep_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                rep_rgb_annot.attach([render_product_path])
                # Manual stepping: do not auto-capture on play.
                try:
                    rep.orchestrator.set_capture_on_play(False)
                except Exception:
                    pass
                _progress(f"[info] Capture backend=replicator, render_product={render_product_path}")
            except Exception as exc:
                _progress(f"[warn] Replicator capture unavailable, falling back to Camera sensor ({type(exc).__name__}: {exc})")
                cap_backend = "camera_sensor"

        if cap_backend not in {"rep", "replicator"}:
            # Fallback: Camera sensor (kept for compatibility).
            try:
                viewport_api = get_active_viewport()
                if viewport_api is not None:
                    render_product_path = viewport_api.get_render_product_path()
                    render_product_source = "viewport"
            except Exception:
                render_product_path = None
                render_product_source = None

            camera = Camera(
                prim_path=cam_prim_path,
                position=cam_pos0,
                orientation=cam_q0,
                resolution=cap_res,
                render_product_path=render_product_path,
            )
        # Improve default camera settings for showcase.
        try:
            cam_prim = stage.GetPrimAtPath("/World/ShowcaseCamera")
            usd_cam = UsdGeom.Camera(cam_prim)
            usd_cam.GetFocalLengthAttr().Set(float(cam_focal_mm))
            usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))
        except Exception:
            pass
        # If we are capturing from the viewport render product, ensure the viewport uses this camera.
        if viewport_api is not None:
            try:
                viewport_api.set_active_camera(cam_prim_path)
            except Exception:
                pass
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()
        if camera is not None:
            camera.initialize()
            # Be explicit in headless mode: attach RGB annotator if not already present.
            try:
                camera.add_rgb_to_frame()
            except Exception:
                pass

        timeseries_path = run.run_dir / "timeseries.jsonl"
        mean_pixels = []
        frames_saved = 0
        screenshot_raw_saved = False
        cam_adjustments = 0
        cam_los_adjustments = 0
        quality_stats = []

        # Warm-up renders (shader compilation, etc).
        for _ in range(max(0, warmup_steps)):
            simulation_app.update()

        # Ensure capture pipeline is alive before entering the main loop.
        if cap_enabled:
            try:
                if cap_backend in {"rep", "replicator"} and rep is not None:
                    # Warm-up a few orchestrator steps.
                    for _ in range(3):
                        rep.orchestrator.step(pause_timeline=False)
                else:
                    _ = _get_valid_rgba(camera, simulation_app.update, max_tries=60)
            except Exception as exc:
                _progress(f"[warn] Showcase capture warmup incomplete; will keep trying in-loop ({type(exc).__name__}: {exc})")

        capture_warned = False
        _progress(f"[info] Starting simulation loop (steps={steps}, warmup_steps={warmup_steps}, screenshot_step={screenshot_step})")
        for step_idx in range(steps):
            t_s = float(step_idx) * float(physics_dt_s)
            phase = 2.0 * math.pi * (t_s / max(1e-9, period_s))
            uav_pos_m = scene_anchor_m + np.array(
                [radius_m * math.cos(phase), radius_m * math.sin(phase), altitude_m], dtype=float
            )
            # Yaw follows tangent direction.
            vx = -radius_m * math.sin(phase)
            vy = radius_m * math.cos(phase)
            yaw = math.atan2(vy, vx)
            q_yaw = _yaw_quat_wxyz_about_up_axis(yaw, stage_up_axis=stage_up_axis)

            uav_pos = _map_pos_m_to_stage_units(
                uav_pos_m, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
            )
            if uav_mode == "usd_asset":
                # Update UAV root prim transform directly.
                if uav_translate_op is not None:
                    uav_translate_op.Set(Gf.Vec3d(float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2])))
                if uav_orient_op is not None:
                    uav_orient_op.Set(
                        Gf.Quatf(float(q_yaw[0]), Gf.Vec3f(float(q_yaw[1]), float(q_yaw[2]), float(q_yaw[3])))
                    )
            else:
                # Procedural UAV parts (world-frame), using yaw rotation for offsets.
                cy, sy = math.cos(yaw), math.sin(yaw)
                R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

                body.set_world_pose(position=uav_pos, orientation=q_yaw)
                arm_x.set_world_pose(position=uav_pos, orientation=q_yaw)
                arm_y.set_world_pose(position=uav_pos, orientation=q_yaw)
                for motor, off in zip(motors, motor_offsets, strict=True):
                    motor_pos = _map_pos_m_to_stage_units(
                        uav_pos_m + (R_yaw @ off),
                        stage_up_axis=stage_up_axis,
                        stage_meters_per_unit=stage_meters_per_unit,
                    )
                    motor.set_world_pose(position=motor_pos, orientation=q_yaw)

            if cam_follow_uav:
                cam_pos_m = uav_pos_m + cam_offset
            else:
                cam_pos_m = scene_anchor_m + np.array(cam_offset, dtype=float)
            cam_pos_m, cam_adjusted = _avoid_aabbs(
                cam_pos_m,
                building_aabbs,
                clearance_m=camera_clearance_m,
                min_height_m=cam_min_height_m,
            )
            if cam_adjusted:
                cam_adjustments += 1
            cam_los_adjusted = False
            if los_check:
                cam_pos_m, cam_los_adjusted = _ensure_line_of_sight(
                    cam_pos_m,
                    uav_pos_m,
                    building_aabbs,
                    push_m=los_push_m,
                    raise_m=los_raise_m,
                    max_iters=los_max_iters,
                    clearance_m=camera_clearance_m,
                    min_height_m=cam_min_height_m,
                )
                if cam_los_adjusted:
                    cam_los_adjustments += 1
            cam_pos = _map_pos_m_to_stage_units(
                cam_pos_m, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
            )
            target_m = uav_pos_m + cam_target_offset if cam_follow_uav else scene_anchor_m + cam_target_offset
            target_pos = _map_pos_m_to_stage_units(
                target_m, stage_up_axis=stage_up_axis, stage_meters_per_unit=stage_meters_per_unit
            )
            cam_q = _look_at_quat_wxyz(cam_pos, target_pos, cam_up_stage)
            if camera is not None:
                camera.set_world_pose(position=cam_pos, orientation=cam_q)
            else:
                # Update USD camera prim directly (Replicator capture path).
                try:
                    cam_prim = stage.GetPrimAtPath(cam_prim_path)
                    xformable = UsdGeom.Xformable(cam_prim)
                    # Translate
                    translate_ops = [
                        op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    ]
                    translate_op = translate_ops[0] if translate_ops else xformable.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
                    # Orient (Quatf expects (real, imaginary vec3))
                    orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
                    orient_op = orient_ops[0] if orient_ops else xformable.AddOrientOp()
                    orient_op.Set(
                        Gf.Quatf(float(cam_q[0]), Gf.Vec3f(float(cam_q[1]), float(cam_q[2]), float(cam_q[3])))
                    )
                except Exception:
                    pass

            simulation_app.update()

            _write_jsonl(
                timeseries_path,
                {
                    "step": int(step_idx),
                    "t_s": float(t_s),
                    "uav_x": float(uav_pos[0]),
                    "uav_y": float(uav_pos[1]),
                    "uav_z": float(uav_pos[2]),
                    "uav_yaw": float(yaw),
                    "cam_x": float(cam_pos[0]),
                    "cam_y": float(cam_pos[1]),
                    "cam_z": float(cam_pos[2]),
                    "cam_adjusted": bool(cam_adjusted),
                    "cam_los_adjusted": bool(cam_los_adjusted),
                    "cap_w": int(cap_res[0]),
                    "cap_h": int(cap_res[1]),
                },
            )

            if cap_enabled and (step_idx % cap_every_n == 0):
                try:
                    if cap_backend in {"rep", "replicator"} and rep is not None and rep_rgb_annot is not None:
                        rep.orchestrator.step(pause_timeline=False)
                        rgb = _normalize_rep_rgb_to_u8(rep_rgb_annot.get_data())
                    else:
                        rgba = _get_valid_rgba(camera, simulation_app.update, max_tries=5)
                        rgb = _normalize_rgb_uint8(rgba)
                    metrics = _frame_quality_metrics(rgb)
                    mean_pixels.append(float(metrics["mean"]))
                    quality_stats.append(metrics)
                    frame_path = frames_dir / f"frame_{step_idx:06d}.png"
                    if _maybe_save_rgb(frame_path, rgb):
                        frames_saved += 1

                    if step_idx == screenshot_step:
                        if _maybe_save_rgb(artifacts_dir / "screenshot_raw.png", rgb):
                            screenshot_raw_saved = True
                except Exception as exc:
                    if not capture_warned:
                        print(f"[warn] Capture failed (will keep trying): {type(exc).__name__}: {exc}")
                        capture_warned = True

        # Sanity: catch “all black” or “all white/blank” outputs.
        sanity = {"ok": True, "reason": None}
        if mean_pixels and (max(mean_pixels) - min(mean_pixels) < 1.0):
            sanity = {"ok": False, "reason": "frames have near-constant mean intensity (likely blank/incorrect camera)"}
        if mean_pixels:
            mean_avg = float(sum(mean_pixels) / max(1, len(mean_pixels)))
            if mean_avg > max_mean:
                sanity = {"ok": False, "reason": "frames are near-white (overexposed/blank)"}
            if mean_avg < min_mean:
                sanity = {"ok": False, "reason": "frames are near-black (underexposed/blank)"}
        if quality_stats:
            avg_std = float(sum(m["std"] for m in quality_stats) / max(1, len(quality_stats)))
            avg_edge = float(sum(m["edge"] for m in quality_stats) / max(1, len(quality_stats)))
            if avg_std < min_std:
                sanity = {"ok": False, "reason": "frames have very low contrast/variance"}
            if avg_edge < min_edge:
                sanity = {"ok": False, "reason": "frames have very low edge content (likely blank or too close-up)"}
        if fail_on_bad_frames and not sanity["ok"]:
            raise RuntimeError(f"Showcase sanity check failed: {sanity['reason']}")

        summary = {
            "mode": "showcase",
            "steps": int(steps),
            "physics_dt_s": float(physics_dt_s),
            "rendering_dt_s": float(rendering_dt_s),
            "experience": experience or None,
            "capture": {
                "enabled": bool(frames_saved > 0),
                "requested": bool(cap_enabled),
                "resolution": [int(cap_res[0]), int(cap_res[1])],
                "every_n_steps": int(cap_every_n),
                "frames_dir": str(frames_dir),
                "screenshot_step": int(screenshot_step),
                "frames_saved": int(frames_saved),
                "screenshot_raw_saved": bool(screenshot_raw_saved),
            },
            "resources": {"active_gpu": int(requested_gpu) if requested_gpu is not None else None, "limit_cpu_threads": int(limit_cpu_threads)},
            "render_product": {"source": render_product_source, "path": render_product_path},
            "capture_backend": str(cap_backend),
            "scene": {
                "stage_usd": stage_usd or None,
                "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)) if stage is not None else None,
                "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)) if stage is not None else None,
                "bbox_stage_units": {
                    "min": [float(x) for x in stage_bbox_min_stage_units] if stage_bbox_min_stage_units is not None else None,
                    "max": [float(x) for x in stage_bbox_max_stage_units] if stage_bbox_max_stage_units is not None else None,
                }
                if stage_bbox_min_stage_units is not None and stage_bbox_max_stage_units is not None
                else None,
                "anchor_m": [float(x) for x in scene_anchor_m] if scene_anchor_m is not None else None,
                "center_m": [float(x) for x in stage_center_m] if stage_center_m is not None else None,
                "extent_m": [float(x) for x in stage_extent_m] if stage_extent_m is not None else None,
                "ground_extent_m": float(ground_extent_m),
                "building_count": int(building_count),
                "building_area_m": float(building_area_m),
                "camera_clearance_m": float(camera_clearance_m),
            },
            "uav": {
                "mode": str(uav_mode),
                "prim_path": str(uav_prim_path),
                "asset_usd": str(uav_asset_path) if uav_asset_path else None,
                "scale": float(uav_scale),
            },
            "uav_path": {"altitude_m": float(altitude_m), "radius_m": float(radius_m), "period_s": float(period_s)},
            "render": {"dome_intensity": float(dome_intensity), "sun_intensity": float(sun_intensity)},
            "camera": {
                "min_height_m": float(cam_min_height_m),
                "adjustments": int(cam_adjustments),
                "los_check": bool(los_check),
                "los_adjustments": int(cam_los_adjustments),
            },
            "mean_pixel": {
                "min": float(min(mean_pixels)) if mean_pixels else None,
                "max": float(max(mean_pixels)) if mean_pixels else None,
                "avg": float(sum(mean_pixels) / max(1, len(mean_pixels))) if mean_pixels else None,
            },
            "quality": {
                "min_mean": float(min_mean),
                "max_mean": float(max_mean),
                "min_std": float(min_std),
                "min_edge": float(min_edge),
                "avg_std": float(sum(m["std"] for m in quality_stats) / max(1, len(quality_stats))) if quality_stats else None,
                "avg_edge": float(sum(m["edge"] for m in quality_stats) / max(1, len(quality_stats))) if quality_stats else None,
            },
            "sanity": sanity,
        }
        write_json(run.run_dir / "summary.json", summary)
        _progress(f"[ok] Wrote summary: {run.run_dir / 'summary.json'}")
        return 0
    except Exception:
        tb = traceback.format_exc()
        try:
            error_log.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        _progress(f"[error] Unhandled exception; traceback written to: {error_log}")
        print(tb, file=sys.stderr, flush=True)
        return 3
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
