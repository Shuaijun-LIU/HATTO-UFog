from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from framework_integration.lib.spec import IsaacReplaySpec, MappingSpec


def _py() -> str:
    return "python"


@dataclass(frozen=True)
class Cmd:
    argv: List[str]
    cwd: Optional[Path] = None
    launcher: Optional[str] = None  # optional prefix, e.g. '"$ISAACSIM_ROOT/python.sh"'

    def shell_line(self) -> str:
        if self.launcher:
            if not self.argv:
                return str(self.launcher)
            return f"{self.launcher} {shlex.join([str(a) for a in self.argv])}"
        return shlex.join([str(a) for a in self.argv])


def framework_export_trajectory_cmd(
    *,
    framework_timeseries_parquet: str,
    output_trajectory_json: str,
    every: int = 2,
    max_points: int = 4000,
) -> Cmd:
    # Run from `HATTO-UFog/Framework` so relative imports/paths match.
    return Cmd(
        argv=[
            _py(),
            "scripts/export_trajectory_json.py",
            "--parquet",
            framework_timeseries_parquet,
            "--output",
            output_trajectory_json,
            "--every",
            str(int(every)),
            "--max-points",
            str(int(max_points)),
        ],
        cwd=Path(__file__).resolve().parents[4] / "HATTO-UFog" / "Framework",
    )


def isaac_replay_trajectory_cmd(
    *,
    trajectory_json: str,
    mapping: MappingSpec,
    replay: IsaacReplaySpec,
) -> Cmd:
    script = Path("framework_integration/scripts/replay_framework_trajectory.py")
    argv: List[str] = [
        str(script),
        "--trajectory_json",
        str(trajectory_json),
        "--output",
        str(replay.output_root),
        "--name",
        str(replay.name),
        "--dt_s",
        str(replay.dt_s),
        "--scale_xy",
        str(mapping.scale_xy),
        "--scale_z",
        str(mapping.scale_z),
        "--x_offset_m",
        str(mapping.x_offset_m),
        "--y_offset_m",
        str(mapping.y_offset_m),
        "--z_offset_m",
        str(mapping.z_offset_m),
    ]
    if replay.max_steps and int(replay.max_steps) > 0:
        argv += ["--max_steps", str(int(replay.max_steps))]
    if replay.headless:
        argv += ["--headless"]
    if replay.capture.enabled:
        argv += ["--capture", "--capture_every_n_steps", str(int(replay.capture.every_n_steps))]
        argv += ["--capture_res", str(int(replay.capture.resolution[0])), str(int(replay.capture.resolution[1]))]
        if replay.capture.strict:
            argv += ["--capture_strict"]
    # Must be executed with Isaac Sim's python.sh; keep launcher in double-quotes for $ISAACSIM_ROOT expansion.
    return Cmd(argv=argv, cwd=Path(__file__).resolve().parents[2], launcher="\"$ISAACSIM_ROOT/python.sh\"")

