from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework_integration.lib.framework_io import parquet_info, required_columns_present


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Framework artifacts for AirSim replay/track (static checks only).")
    parser.add_argument("--framework_timeseries", required=True, help="Path to Framework timeseries.parquet")
    parser.add_argument("--mode", default="replay", choices=["replay", "track"], help="Validation mode (replay is less strict).")
    args = parser.parse_args()

    ts = Path(args.framework_timeseries).expanduser().resolve()
    if not ts.exists():
        print(f"[ERROR] Missing file: {ts}")
        return 2
    if ts.suffix.lower() != ".parquet":
        _warn(f"Not a .parquet file: {ts.name}")

    info = parquet_info(ts)
    _ok(f"timeseries={info.path} rows={info.num_rows} cols={len(info.columns)}")

    required: List[str] = ["x", "y"]
    if args.mode == "track":
        required += ["z"]
    ok, missing = required_columns_present(info.columns, required)
    if not ok:
        print(f"[ERROR] Missing required columns: {missing}")
        return 2
    _ok(f"Required columns present: {required}")

    # Optional checks / warnings
    metric_keys = ["S", "E_total", "D_total"]
    present_metrics = [k for k in metric_keys if k in info.columns]
    if not present_metrics:
        _warn(
            "No paper-metric columns found (expected at least one of: "
            + ", ".join(metric_keys)
            + "). Replays/tracking will still work, but only pose/visuals will be available unless metrics are computed upstream (Framework)."
        )
    else:
        _ok(f"Found metric columns (optional): {present_metrics}")

    if "yaw" in info.columns:
        import pyarrow.parquet as pq

        table = pq.read_table(str(ts), columns=["yaw"])
        col = table.column(0)
        sample = col.to_pylist()[:200]
        yaws = [float(v) for v in sample if v is not None]
        if not yaws:
            _warn("`yaw` column exists but sample has no non-null values; `--use_yaw` / yaw tracking may not be meaningful.")
        else:
            max_abs = max(abs(v) for v in yaws)
            # If yaw seems like degrees, it will typically exceed 2*pi.
            if max_abs > 2.0 * math.pi + 0.6:
                _warn(f"`yaw` max|val|≈{max_abs:.2f} looks > 2π; check if degrees vs radians.")
            else:
                _ok("`yaw` looks like radians (heuristic).")

    run_dir = ts.parent
    for name in ["config.json", "summary.json", "world.json"]:
        p = run_dir / name
        if p.exists():
            _ok(f"Found optional artifact: {p}")
        else:
            _warn(f"Optional artifact not found (fine for replay): {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
