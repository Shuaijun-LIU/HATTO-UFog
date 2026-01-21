from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework_integration.lib.commands import framework_export_trajectory_cmd, isaac_replay_trajectory_cmd
from framework_integration.lib.spec import (
    ExportTrajectorySpec,
    FrameworkSpec,
    IsaacReplaySpec,
    MappingSpec,
    PipelineSpec,
    load_pipeline_spec,
)


def _write_sh(path: Path, *, lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + lines.strip() + "\n", encoding="utf-8")


def _coalesce_pipeline(args) -> PipelineSpec:
    if args.spec:
        return load_pipeline_spec(args.spec)
    pipeline = PipelineSpec(
        framework=FrameworkSpec(
            trajectory_json=str(args.trajectory_json) if args.trajectory_json else None,
            timeseries_parquet=str(args.framework_timeseries) if args.framework_timeseries else None,
        ),
        export_trajectory=ExportTrajectorySpec(every=int(args.export_every), max_points=int(args.export_max_points)),
        mapping=MappingSpec(
            scale_xy=float(args.scale_xy),
            scale_z=float(args.scale_z),
            x_offset_m=float(args.x_offset_m),
            y_offset_m=float(args.y_offset_m),
            z_offset_m=float(args.z_offset_m),
        ),
        isaac_replay=IsaacReplaySpec(
            headless=bool(args.headless),
            dt_s=float(args.dt_s),
            max_steps=int(args.max_steps),
            output_root=str(args.output_root),
            name=str(args.name),
        ),
    )
    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a planning pack (commands + meta placeholders) for Framework→Isaac replay. Does not execute."
    )
    parser.add_argument("--spec", default="", help="Optional pipeline YAML spec. If provided, overrides individual flags.")
    parser.add_argument("--framework_timeseries", default="", help="Path to Framework timeseries.parquet (for planning export).")
    parser.add_argument("--trajectory_json", default="", help="Existing trajectory.json path (skip export planning).")

    parser.add_argument("--plan_output_root", default="framework_integration/plans_isaac", help="Where to create plan packs (under Isaac/).")
    parser.add_argument("--name", default="fw_replay", help="Run name (used for plan pack id and default replay name).")

    # Export knobs (used if planning trajectory export)
    parser.add_argument("--export_every", type=int, default=2)
    parser.add_argument("--export_max_points", type=int, default=4000)

    # Mapping knobs
    parser.add_argument("--scale_xy", type=float, default=1.0)
    parser.add_argument("--scale_z", type=float, default=1.0)
    parser.add_argument("--x_offset_m", type=float, default=0.0)
    parser.add_argument("--y_offset_m", type=float, default=0.0)
    parser.add_argument("--z_offset_m", type=float, default=0.0)

    # Replay knobs
    parser.add_argument("--headless", action="store_true", help="Plan headless replay.")
    parser.add_argument("--dt_s", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--output_root", default="runs_isaac")

    args = parser.parse_args()
    pipeline = _coalesce_pipeline(args)

    if not pipeline.framework.trajectory_json and not pipeline.framework.timeseries_parquet:
        raise SystemExit("[error] Provide --trajectory_json or --framework_timeseries (or use --spec).")

    from bridge.runpack import prepare_run, write_json

    base_dir = Path(__file__).resolve().parents[2]  # .../Isaac
    plan_output_root = (base_dir / args.plan_output_root).resolve()
    ctx = prepare_run(output_root=plan_output_root, name=str(args.name), extra_meta={"kind": "framework_integration_plan"})

    # Resolve planned trajectory output path.
    planned_inputs_dir = ctx.run_dir / "inputs"
    planned_inputs_dir.mkdir(parents=True, exist_ok=True)
    planned_traj_json = planned_inputs_dir / "trajectory.json"
    traj_json_path = pipeline.framework.trajectory_json or str(planned_traj_json)

    write_json(
        ctx.run_dir / "plan.json",
        {
            "framework_timeseries": pipeline.framework.timeseries_parquet,
            "trajectory_json": pipeline.framework.trajectory_json,
            "planned_trajectory_json": str(planned_traj_json),
            "export_trajectory": pipeline.export_trajectory.__dict__,
            "mapping": pipeline.mapping.__dict__,
            "isaac_replay": {
                "headless": pipeline.isaac_replay.headless,
                "dt_s": pipeline.isaac_replay.dt_s,
                "max_steps": pipeline.isaac_replay.max_steps,
                "output_root": pipeline.isaac_replay.output_root,
                "name": pipeline.isaac_replay.name,
            },
        },
    )

    if pipeline.framework.timeseries_parquet and not pipeline.framework.trajectory_json:
        export_cmd = framework_export_trajectory_cmd(
            framework_timeseries_parquet=str(pipeline.framework.timeseries_parquet),
            output_trajectory_json=str(planned_traj_json),
            every=int(pipeline.export_trajectory.every),
            max_points=int(pipeline.export_trajectory.max_points),
        )
        _write_sh(ctx.run_dir / "planned_framework_export_trajectory.sh", lines=f"cd {export_cmd.cwd}\n{export_cmd.shell_line()}\n")

    replay_cmd = isaac_replay_trajectory_cmd(trajectory_json=traj_json_path, mapping=pipeline.mapping, replay=pipeline.isaac_replay)
    _write_sh(ctx.run_dir / "planned_isaac_replay_trajectory.sh", lines=f"cd {replay_cmd.cwd}\n{replay_cmd.shell_line()}\n")

    print(f"Plan pack: {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
