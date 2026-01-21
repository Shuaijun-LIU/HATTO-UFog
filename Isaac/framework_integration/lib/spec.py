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
class FrameworkSpec:
    trajectory_json: Optional[str] = None
    timeseries_parquet: Optional[str] = None


@dataclass(frozen=True)
class ExportTrajectorySpec:
    every: int = 2
    max_points: int = 4000


@dataclass(frozen=True)
class MappingSpec:
    scale_xy: float = 1.0
    scale_z: float = 1.0
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    z_offset_m: float = 0.0


@dataclass(frozen=True)
class CaptureSpec:
    enabled: bool = False
    every_n_steps: int = 5
    resolution: tuple[int, int] = (640, 360)
    strict: bool = False


@dataclass(frozen=True)
class IsaacReplaySpec:
    headless: bool = True
    dt_s: float = 0.05
    max_steps: int = 0
    output_root: str = "runs_isaac"
    name: str = "fw_replay"
    capture: CaptureSpec = CaptureSpec()


@dataclass(frozen=True)
class PipelineSpec:
    framework: FrameworkSpec
    export_trajectory: ExportTrajectorySpec = ExportTrajectorySpec()
    mapping: MappingSpec = MappingSpec()
    isaac_replay: IsaacReplaySpec = IsaacReplaySpec()


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline spec YAML must be a mapping at the root.")

    fw_d = _as_dict(data.get("framework"))
    export_d = _as_dict(data.get("export_trajectory"))
    mapping_d = _as_dict(data.get("mapping"))
    replay_d = _as_dict(data.get("isaac_replay"))
    capture_d = _as_dict(replay_d.get("capture"))

    fw = FrameworkSpec(
        trajectory_json=str(fw_d["trajectory_json"]) if fw_d.get("trajectory_json") else None,
        timeseries_parquet=str(fw_d["timeseries_parquet"]) if fw_d.get("timeseries_parquet") else None,
    )
    if not fw.trajectory_json and not fw.timeseries_parquet:
        raise ValueError("pipeline.framework.trajectory_json or pipeline.framework.timeseries_parquet is required.")

    export = ExportTrajectorySpec(
        every=int(export_d.get("every", 2)),
        max_points=int(export_d.get("max_points", 4000)),
    )

    mapping = MappingSpec(
        scale_xy=float(mapping_d.get("scale_xy", 1.0)),
        scale_z=float(mapping_d.get("scale_z", 1.0)),
        x_offset_m=float(mapping_d.get("x_offset_m", 0.0)),
        y_offset_m=float(mapping_d.get("y_offset_m", 0.0)),
        z_offset_m=float(mapping_d.get("z_offset_m", 0.0)),
    )

    cap_res = capture_d.get("resolution", [640, 360])
    try:
        cap_res_t = (int(cap_res[0]), int(cap_res[1]))
    except Exception:
        cap_res_t = (640, 360)
    capture = CaptureSpec(
        enabled=bool(capture_d.get("enabled", False)),
        every_n_steps=int(capture_d.get("every_n_steps", 5)),
        resolution=cap_res_t,
        strict=bool(capture_d.get("strict", False)),
    )

    replay = IsaacReplaySpec(
        headless=bool(replay_d.get("headless", True)),
        dt_s=float(replay_d.get("dt_s", 0.05)),
        max_steps=int(replay_d.get("max_steps", 0)),
        output_root=str(replay_d.get("output_root", "runs_isaac")),
        name=str(replay_d.get("name", "fw_replay")),
        capture=capture,
    )

    return PipelineSpec(framework=fw, export_trajectory=export, mapping=mapping, isaac_replay=replay)

