"""Run a single baseline experiment."""
from __future__ import annotations

import argparse
import json

from ufog_network.config import load_config
from ufog_network.sim.simulator import Simulator
from ufog_network.io import import_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config yaml/json")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--world", default=None, help="Optional world.json to load")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world = import_world(args.world) if args.world else None
    sim = Simulator(cfg, world=world)
    result = sim.run(output_root=args.output)
    print(json.dumps(result.summary, indent=2))
    print(f"Output: {result.output_dir}")


if __name__ == "__main__":
    main()
