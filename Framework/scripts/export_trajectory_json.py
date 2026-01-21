"""Export a lightweight trajectory JSON from a run's Parquet timeseries.

This is useful for visualization/replay (e.g., the showcase viewer).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True, help="Path to timeseries.parquet")
    parser.add_argument("--output", type=str, required=True, help="Output trajectory JSON path")
    parser.add_argument("--every", type=int, default=2, help="Keep every N rows (downsampling)")
    parser.add_argument("--max-points", type=int, default=4000, help="Cap output points")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet, columns=["time_s", "x", "y", "z"])
    every = max(1, int(args.every))
    df = df.iloc[::every].copy()
    if len(df) > int(args.max_points):
        df = df.iloc[: int(args.max_points)].copy()

    traj = {
        "meta": {
            "source": str(Path(args.parquet).as_posix()),
            "units": "m",
            "coord": "sim(x,y,z) with z up",
            "points": int(len(df)),
            "time_s_min": float(df["time_s"].min()) if len(df) else 0.0,
            "time_s_max": float(df["time_s"].max()) if len(df) else 0.0,
            "downsample_every": every,
        },
        "path": df[["x", "y", "z"]].astype(float).values.tolist(),
        "time_s": df["time_s"].astype(float).values.tolist(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(traj, indent=2))
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
