from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Trajectory:
    path_xyz: List[Tuple[float, float, float]]
    time_s: Optional[List[float]] = None
    yaw_rad: Optional[List[float]] = None
    rpy_rad: Optional[List[Tuple[float, float, float]]] = None
    meta: Optional[Dict[str, Any]] = None


def load_trajectory_json(path: str | Path) -> Trajectory:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("trajectory.json must be a dict at root.")

    raw_path = data.get("path")
    if not isinstance(raw_path, list) or not raw_path:
        raise ValueError("trajectory.json.path must be a non-empty list of [x,y,z].")

    path_xyz: List[Tuple[float, float, float]] = []
    for i, row in enumerate(raw_path):
        if not (isinstance(row, list) or isinstance(row, tuple)) or len(row) < 3:
            raise ValueError(f"trajectory.json.path[{i}] must be [x,y,z].")
        path_xyz.append((float(row[0]), float(row[1]), float(row[2])))

    time_s = data.get("time_s")
    if time_s is not None:
        if not isinstance(time_s, list) or len(time_s) != len(path_xyz):
            raise ValueError("trajectory.json.time_s must be a list with the same length as path.")
        time_s = [float(t) for t in time_s]

    yaw = data.get("yaw_rad")
    if yaw is not None:
        if not isinstance(yaw, list) or len(yaw) != len(path_xyz):
            raise ValueError("trajectory.json.yaw_rad must be a list with the same length as path.")
        yaw = [float(v) for v in yaw]

    rpy = data.get("rpy_rad")
    rpy_rad: Optional[List[Tuple[float, float, float]]] = None
    if rpy is not None:
        if not isinstance(rpy, list) or len(rpy) != len(path_xyz):
            raise ValueError("trajectory.json.rpy_rad must be a list with the same length as path.")
        tmp: List[Tuple[float, float, float]] = []
        for i, row in enumerate(rpy):
            if not (isinstance(row, list) or isinstance(row, tuple)) or len(row) < 3:
                raise ValueError(f"trajectory.json.rpy_rad[{i}] must be [roll,pitch,yaw].")
            tmp.append((float(row[0]), float(row[1]), float(row[2])))
        rpy_rad = tmp

    meta = data.get("meta")
    if meta is not None and not isinstance(meta, dict):
        meta = {"_meta_parse_warning": "meta was not a dict"}

    return Trajectory(path_xyz=path_xyz, time_s=time_s, yaw_rad=yaw, rpy_rad=rpy_rad, meta=meta if isinstance(meta, dict) else None)

