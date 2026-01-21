from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework_integration.lib.spec import load_pipeline_spec
from framework_integration.lib.trajectory_io import load_trajectory_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Framework→Isaac integration inputs (no Isaac required).")
    parser.add_argument("--spec", default="", help="Optional pipeline YAML spec.")
    parser.add_argument("--trajectory_json", default="", help="Trajectory JSON path (overrides spec).")
    parser.add_argument("--framework_timeseries", default="", help="Framework timeseries.parquet path (planning-only; overrides spec).")
    parser.add_argument("--require_trajectory", action="store_true", help="Fail if trajectory JSON is not provided/resolved.")
    args = parser.parse_args()

    traj_path = args.trajectory_json.strip()
    ts_path = args.framework_timeseries.strip()
    if args.spec:
        spec = load_pipeline_spec(args.spec)
        if not traj_path and spec.framework.trajectory_json:
            traj_path = spec.framework.trajectory_json
        if not ts_path and spec.framework.timeseries_parquet:
            ts_path = spec.framework.timeseries_parquet

    if ts_path:
        p = Path(ts_path).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"[error] timeseries not found: {p}")
        if p.suffix.lower() != ".parquet":
            raise SystemExit(f"[error] timeseries must be a .parquet file: {p}")
        print(f"[ok] timeseries: {p}")

    if traj_path:
        p = Path(traj_path).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"[error] trajectory not found: {p}")
        traj = load_trajectory_json(p)
        if len(traj.path_xyz) < 2:
            raise SystemExit("[error] trajectory must contain at least 2 points.")
        if traj.time_s is not None and len(traj.time_s) != len(traj.path_xyz):
            raise SystemExit("[error] trajectory time_s length mismatch.")
        if traj.yaw_rad is not None and len(traj.yaw_rad) != len(traj.path_xyz):
            raise SystemExit("[error] trajectory yaw_rad length mismatch.")
        if traj.rpy_rad is not None and len(traj.rpy_rad) != len(traj.path_xyz):
            raise SystemExit("[error] trajectory rpy_rad length mismatch.")
        print(f"[ok] trajectory: {p} (points={len(traj.path_xyz)})")
    elif args.require_trajectory:
        raise SystemExit("[error] trajectory_json is required (provide --trajectory_json or spec.framework.trajectory_json).")
    else:
        print("[warn] no trajectory_json provided; only checked timeseries path (if any).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

