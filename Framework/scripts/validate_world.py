"""Validate world generation and determinism fingerprint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufog_network.config import load_config
from ufog_network.env.generators import generate_world
from ufog_network.utils.validation import validate_world, validate_waypoint_edges, world_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config yaml/json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world = generate_world(cfg.world)
    checks = validate_world(world)
    edge_checks = validate_waypoint_edges(world)
    fingerprint = world_fingerprint(world)
    ok = all(checks.values()) and all(edge_checks.values())
    print({"checks": checks, "edges": edge_checks, "fingerprint": fingerprint})
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
