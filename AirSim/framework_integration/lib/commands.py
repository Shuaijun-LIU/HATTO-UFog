from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from framework_integration.lib.spec import MappingSpec, ReplayMainlineSpec, TrackAuxlineSpec


def _py() -> str:
    return "python"


@dataclass(frozen=True)
class Cmd:
    argv: List[str]
    cwd: Optional[Path] = None

    def shell_line(self) -> str:
        return shlex.join([str(a) for a in self.argv])


def framework_run_cmd(*, framework_config: str, framework_output_root: str = "runs") -> Cmd:
    # Prefer the module form (works even if entrypoint script isn't installed),
    # assuming the user runs it from `HATTO-UFog/Framework`.
    return Cmd(
        argv=[_py(), "-m", "ufog_network.cli", "run", "--config", framework_config, "--output", framework_output_root],
        cwd=Path(__file__).resolve().parents[4] / "HATTO-UFog" / "Framework",
    )


def airsim_replay_mainline_plus_cmd(
    *,
    framework_timeseries: str,
    mapping: MappingSpec,
    replay: ReplayMainlineSpec,
    output_root: str = "runs_airsim",
    settings_template: str = "configs/airsim_settings/settings_mainline.json",
) -> Cmd:
    script = Path("framework_integration/scripts/replay_framework_mainline_plus.py")
    argv: List[str] = [
        _py(),
        str(script),
        "--framework_timeseries",
        framework_timeseries,
        "--output_root",
        output_root,
        "--settings_template",
        settings_template,
        "--dt",
        str(replay.dt),
        "--fps",
        str(replay.fps),
        "--stride",
        str(replay.stride),
    ]
    if replay.max_steps and int(replay.max_steps) > 0:
        argv += ["--max_steps", str(int(replay.max_steps))]
    argv += ["--scale_xy", str(mapping.scale_xy), "--scale_z", str(mapping.scale_z)]
    argv += ["--x_offset_ned", str(mapping.x_offset_ned), "--y_offset_ned", str(mapping.y_offset_ned), "--z_offset_ned", str(mapping.z_offset_ned)]
    if mapping.z_up_m is not None:
        argv += ["--z_up_m", str(mapping.z_up_m)]
    else:
        argv += ["--base_z_up_m", str(mapping.base_z_up_m)]
    if mapping.auto_offset:
        argv += ["--auto_offset"]
    if replay.overlay:
        argv += ["--overlay"]
    if replay.use_yaw:
        argv += ["--use_yaw"]
    if replay.use_rpy:
        argv += ["--use_rpy"]
    if replay.ignore_collision:
        argv += ["--ignore_collision"]
    # Run from AirSim folder so relative paths match existing scripts.
    return Cmd(argv=argv, cwd=Path(__file__).resolve().parents[2])


def airsim_track_auxline_cmd(
    *,
    framework_timeseries: str,
    mapping: MappingSpec,
    track: TrackAuxlineSpec,
    output_root: str = "runs_airsim",
    settings_template: str = "configs/airsim_settings/settings_auxline.json",
) -> Cmd:
    script = Path("framework_integration/scripts/track_framework_auxline.py")
    argv: List[str] = [
        _py(),
        str(script),
        "--framework_timeseries",
        framework_timeseries,
        "--output_root",
        output_root,
        "--settings_template",
        settings_template,
        "--dt",
        str(track.dt),
        "--fps",
        str(track.fps),
        "--stride",
        str(track.stride),
        "--speed_m_s",
        str(track.speed_m_s),
    ]
    if track.max_steps and int(track.max_steps) > 0:
        argv += ["--max_steps", str(int(track.max_steps))]
    argv += ["--scale_xy", str(mapping.scale_xy), "--scale_z", str(mapping.scale_z)]
    argv += ["--x_offset_ned", str(mapping.x_offset_ned), "--y_offset_ned", str(mapping.y_offset_ned), "--z_offset_ned", str(mapping.z_offset_ned)]
    if mapping.z_up_m is not None:
        argv += ["--z_up_m", str(mapping.z_up_m)]
    else:
        argv += ["--base_z_up_m", str(mapping.base_z_up_m)]
    if track.overlay:
        argv += ["--overlay"]
    argv += ["--yaw_mode", str(track.yaw_mode)]
    argv += ["--fixed_yaw_deg", str(track.fixed_yaw_deg)]
    # Run from AirSim folder so relative paths match existing scripts.
    return Cmd(argv=argv, cwd=Path(__file__).resolve().parents[2])
