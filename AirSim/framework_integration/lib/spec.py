from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dict-like YAML node, got: {type(obj).__name__}")


@dataclass(frozen=True)
class MappingSpec:
    scale_xy: float = 1.0
    scale_z: float = 1.0
    x_offset_ned: float = 0.0
    y_offset_ned: float = 0.0
    z_offset_ned: float = 0.0
    auto_offset: bool = False
    base_z_up_m: float = 10.0
    z_up_m: Optional[float] = None


@dataclass(frozen=True)
class ReplayMainlineSpec:
    dt: float = 0.05
    fps: float = 20.0
    stride: int = 1
    max_steps: int = 0
    overlay: bool = True
    use_yaw: bool = True
    use_rpy: bool = False
    ignore_collision: bool = False


@dataclass(frozen=True)
class TrackAuxlineSpec:
    dt: float = 0.05
    fps: float = 20.0
    stride: int = 1
    max_steps: int = 0
    speed_m_s: float = 3.0
    overlay: bool = True
    yaw_mode: str = "face_path"  # face_path | from_timeseries | fixed
    fixed_yaw_deg: float = 0.0


@dataclass(frozen=True)
class FrameworkSpec:
    timeseries: str


@dataclass(frozen=True)
class PipelineSpec:
    framework: FrameworkSpec
    mapping: MappingSpec = MappingSpec()
    replay_mainline: ReplayMainlineSpec = ReplayMainlineSpec()
    track_auxline: TrackAuxlineSpec = TrackAuxlineSpec()


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline spec YAML must be a mapping at the root.")

    fw = _as_dict(data.get("framework"))
    timeseries = fw.get("timeseries")
    if not timeseries:
        raise ValueError("pipeline.framework.timeseries is required.")

    mapping_d = _as_dict(data.get("mapping"))
    replay_d = _as_dict(data.get("replay_mainline"))
    track_d = _as_dict(data.get("track_auxline"))

    mapping = MappingSpec(
        scale_xy=float(mapping_d.get("scale_xy", 1.0)),
        scale_z=float(mapping_d.get("scale_z", 1.0)),
        x_offset_ned=float(mapping_d.get("x_offset_ned", 0.0)),
        y_offset_ned=float(mapping_d.get("y_offset_ned", 0.0)),
        z_offset_ned=float(mapping_d.get("z_offset_ned", 0.0)),
        auto_offset=bool(mapping_d.get("auto_offset", False)),
        base_z_up_m=float(mapping_d.get("base_z_up_m", 10.0)),
        z_up_m=float(mapping_d["z_up_m"]) if "z_up_m" in mapping_d and mapping_d["z_up_m"] is not None else None,
    )

    replay = ReplayMainlineSpec(
        dt=float(replay_d.get("dt", 0.05)),
        fps=float(replay_d.get("fps", 20.0)),
        stride=int(replay_d.get("stride", 1)),
        max_steps=int(replay_d.get("max_steps", 0)),
        overlay=bool(replay_d.get("overlay", True)),
        use_yaw=bool(replay_d.get("use_yaw", True)),
        use_rpy=bool(replay_d.get("use_rpy", False)),
        ignore_collision=bool(replay_d.get("ignore_collision", False)),
    )

    track = TrackAuxlineSpec(
        dt=float(track_d.get("dt", 0.05)),
        fps=float(track_d.get("fps", 20.0)),
        stride=int(track_d.get("stride", 1)),
        max_steps=int(track_d.get("max_steps", 0)),
        speed_m_s=float(track_d.get("speed_m_s", 3.0)),
        overlay=bool(track_d.get("overlay", True)),
        yaw_mode=str(track_d.get("yaw_mode", "face_path")),
        fixed_yaw_deg=float(track_d.get("fixed_yaw_deg", 0.0)),
    )

    return PipelineSpec(
        framework=FrameworkSpec(timeseries=str(timeseries)),
        mapping=mapping,
        replay_mainline=replay,
        track_auxline=track,
    )

