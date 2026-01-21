from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework_integration.lib.commands import airsim_replay_mainline_plus_cmd, airsim_track_auxline_cmd, framework_run_cmd
from framework_integration.lib.spec import MappingSpec, ReplayMainlineSpec, TrackAuxlineSpec, load_pipeline_spec


def _write_sh(path: Path, *, lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + lines.strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a planning pack (commands + meta placeholders) for Framework→AirSim validation. Does not execute."
    )
    parser.add_argument("--spec", default="", help="Optional pipeline YAML spec. If provided, overrides individual flags.")
    parser.add_argument("--framework_timeseries", default="", help="Path to Framework timeseries.parquet (for replay/track planning).")
    parser.add_argument("--framework_config", default="", help="Optional: Framework config yaml/json (for planning a Framework run command).")
    parser.add_argument("--framework_output_root", default="runs", help="Framework output dir (used only if framework_config is set).")

    parser.add_argument("--plan_output_root", default="plans_airsim", help="Where to create plan packs (under AirSim repo).")
    parser.add_argument("--name", default="framework_integration_plan", help="Plan pack name (used for run_id).")

    # Mapping (subset; can be set by spec)
    parser.add_argument("--scale_xy", type=float, default=1.0)
    parser.add_argument("--scale_z", type=float, default=1.0)
    parser.add_argument("--x_offset_ned", type=float, default=0.0)
    parser.add_argument("--y_offset_ned", type=float, default=0.0)
    parser.add_argument("--z_offset_ned", type=float, default=0.0)
    parser.add_argument("--auto_offset", action="store_true")
    parser.add_argument("--base_z_up_m", type=float, default=10.0)
    parser.add_argument("--z_up_m", type=float, default=None)

    # Replay knobs
    parser.add_argument("--replay_dt", type=float, default=0.05)
    parser.add_argument("--replay_fps", type=float, default=20.0)
    parser.add_argument("--replay_stride", type=int, default=1)
    parser.add_argument("--replay_max_steps", type=int, default=0)
    parser.add_argument("--replay_overlay", action="store_true")
    parser.add_argument("--replay_use_yaw", action="store_true")
    parser.add_argument("--replay_use_rpy", action="store_true")

    # Track knobs
    parser.add_argument("--track_dt", type=float, default=0.05)
    parser.add_argument("--track_fps", type=float, default=20.0)
    parser.add_argument("--track_stride", type=int, default=1)
    parser.add_argument("--track_max_steps", type=int, default=0)
    parser.add_argument("--track_speed_m_s", type=float, default=3.0)
    parser.add_argument("--track_overlay", action="store_true")
    parser.add_argument("--track_yaw_mode", default="face_path", choices=["face_path", "from_timeseries", "fixed"])
    parser.add_argument("--track_fixed_yaw_deg", type=float, default=0.0)

    args = parser.parse_args()

    if args.spec:
        spec = load_pipeline_spec(args.spec)
        framework_timeseries = spec.framework.timeseries
        mapping = spec.mapping
        replay = spec.replay_mainline
        track = spec.track_auxline
    else:
        framework_timeseries = args.framework_timeseries
        mapping = MappingSpec(
            scale_xy=float(args.scale_xy),
            scale_z=float(args.scale_z),
            x_offset_ned=float(args.x_offset_ned),
            y_offset_ned=float(args.y_offset_ned),
            z_offset_ned=float(args.z_offset_ned),
            auto_offset=bool(args.auto_offset),
            base_z_up_m=float(args.base_z_up_m),
            z_up_m=float(args.z_up_m) if args.z_up_m is not None else None,
        )
        replay = ReplayMainlineSpec(
            dt=float(args.replay_dt),
            fps=float(args.replay_fps),
            stride=int(args.replay_stride),
            max_steps=int(args.replay_max_steps),
            overlay=bool(args.replay_overlay),
            use_yaw=bool(args.replay_use_yaw),
            use_rpy=bool(args.replay_use_rpy),
        )
        track = TrackAuxlineSpec(
            dt=float(args.track_dt),
            fps=float(args.track_fps),
            stride=int(args.track_stride),
            max_steps=int(args.track_max_steps),
            speed_m_s=float(args.track_speed_m_s),
            overlay=bool(args.track_overlay),
            yaw_mode=str(args.track_yaw_mode),
            fixed_yaw_deg=float(args.track_fixed_yaw_deg),
        )

    # Use AirSim bridge runpack to keep consistent metadata layout.
    from bridge.runpack import prepare_run, write_json

    base_dir = Path(__file__).resolve().parents[2]  # .../AirSim
    plan_output_root = (base_dir / args.plan_output_root).resolve()
    ctx = prepare_run(output_root=plan_output_root, name=str(args.name), extra_meta={"kind": "framework_integration_plan"})

    write_json(
        ctx.run_dir / "plan.json",
        {
            "framework_timeseries": framework_timeseries or None,
            "framework_config": args.framework_config or None,
            "framework_output_root": args.framework_output_root,
            "mapping": mapping.__dict__,
            "replay_mainline": replay.__dict__,
            "track_auxline": track.__dict__,
        },
    )

    if args.framework_config:
        fw_cmd = framework_run_cmd(framework_config=str(args.framework_config), framework_output_root=str(args.framework_output_root))
        _write_sh(ctx.run_dir / "planned_framework_run.sh", lines=f"cd {fw_cmd.cwd}\n{fw_cmd.shell_line()}\n")

    if framework_timeseries:
        replay_cmd = airsim_replay_mainline_plus_cmd(
            framework_timeseries=str(framework_timeseries),
            mapping=mapping,
            replay=replay,
            output_root="runs_airsim",
            settings_template="configs/airsim_settings/settings_mainline.json",
        )
        _write_sh(ctx.run_dir / "planned_airsim_replay_mainline.sh", lines=f"cd {replay_cmd.cwd}\n{replay_cmd.shell_line()}\n")

        track_cmd = airsim_track_auxline_cmd(
            framework_timeseries=str(framework_timeseries),
            mapping=mapping,
            track=track,
            output_root="runs_airsim",
            settings_template="configs/airsim_settings/settings_auxline.json",
        )
        _write_sh(ctx.run_dir / "planned_airsim_track_auxline.sh", lines=f"cd {track_cmd.cwd}\n{track_cmd.shell_line()}\n")

    print(f"Plan pack: {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
