from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework_integration.lib.commands import airsim_track_auxline_cmd
from framework_integration.lib.spec import MappingSpec, TrackAuxlineSpec


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan (print) an AirSim auxline tracking command for a Framework timeseries.")
    parser.add_argument("--framework_timeseries", required=True)
    parser.add_argument("--output_root", default="runs_airsim")
    parser.add_argument("--settings_template", default="configs/airsim_settings/settings_auxline.json")

    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--speed_m_s", type=float, default=3.0)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--yaw_mode", default="face_path", choices=["face_path", "from_timeseries", "fixed"])
    parser.add_argument("--fixed_yaw_deg", type=float, default=0.0)

    parser.add_argument("--scale_xy", type=float, default=1.0)
    parser.add_argument("--scale_z", type=float, default=1.0)
    parser.add_argument("--x_offset_ned", type=float, default=0.0)
    parser.add_argument("--y_offset_ned", type=float, default=0.0)
    parser.add_argument("--z_offset_ned", type=float, default=0.0)
    parser.add_argument("--base_z_up_m", type=float, default=10.0)
    parser.add_argument("--z_up_m", type=float, default=None)

    parser.add_argument("--write", default="", help="If set, write the command into this .sh file (relative to CWD).")
    args = parser.parse_args()

    mapping = MappingSpec(
        scale_xy=float(args.scale_xy),
        scale_z=float(args.scale_z),
        x_offset_ned=float(args.x_offset_ned),
        y_offset_ned=float(args.y_offset_ned),
        z_offset_ned=float(args.z_offset_ned),
        auto_offset=False,
        base_z_up_m=float(args.base_z_up_m),
        z_up_m=float(args.z_up_m) if args.z_up_m is not None else None,
    )
    track = TrackAuxlineSpec(
        dt=float(args.dt),
        fps=float(args.fps),
        stride=int(args.stride),
        max_steps=int(args.max_steps),
        speed_m_s=float(args.speed_m_s),
        overlay=bool(args.overlay),
        yaw_mode=str(args.yaw_mode),
        fixed_yaw_deg=float(args.fixed_yaw_deg),
    )

    cmd = airsim_track_auxline_cmd(
        framework_timeseries=str(args.framework_timeseries),
        mapping=mapping,
        track=track,
        output_root=str(args.output_root),
        settings_template=str(args.settings_template),
    )
    line = cmd.shell_line()
    print(line)

    if args.write:
        p = Path(args.write).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + line + "\n", encoding="utf-8")
        print(f"Wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
