"""Run a short simulation and validate outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufog_network.config import load_config
from ufog_network.sim.simulator import Simulator
from ufog_network.utils.validation import validate_timeseries_schema, schema_missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config yaml/json")
    parser.add_argument("--output", default="runs/validate", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.sim.steps = 10
    sim = Simulator(cfg)
    result = sim.run(output_root=args.output)
    parquet_path = Path(result.output_dir) / "timeseries.parquet"
    schema_ok = validate_timeseries_schema(str(parquet_path))
    missing = schema_missing(schema_ok)
    if missing:
        print(f"Missing columns: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
