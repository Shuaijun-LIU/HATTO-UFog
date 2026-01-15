"""Analyze MD distance to nearest waypoint nodes."""
from __future__ import annotations

import argparse
import math

import numpy as np

from ufog_network.config import load_config
from ufog_network.env.generators import generate_world
from ufog_network.env.tasks import generate_md_positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config yaml/json")
    parser.add_argument("--sample", type=int, default=0, help="Sample count to print (0 disables)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world = generate_world(cfg.world)
    md_positions = generate_md_positions(world, cfg.tasks, cfg.sim.seed)

    if not world.waypoints or not world.waypoints.nodes:
        print({"error": "no_waypoints", "md_count": len(md_positions)})
        return 1

    nodes = world.waypoints.nodes
    dists = []
    los_count = 0
    for pos in md_positions:
        best_d2 = 1e18
        best_node = None
        for n in nodes:
            dx = n[0] - pos[0]
            dy = n[1] - pos[1]
            dz = n[2] - pos[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_node = n
        dist = math.sqrt(best_d2)
        dists.append(dist)
        if best_node and world.segment_is_free(pos, best_node, step=world.cfg.connect_step_m):
            los_count += 1

    arr = np.array(dists, dtype=float)
    stats = {
        "md_count": int(len(md_positions)),
        "min_dist_m": float(arr.min()) if len(arr) else 0.0,
        "p25_dist_m": float(np.percentile(arr, 25)) if len(arr) else 0.0,
        "median_dist_m": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p75_dist_m": float(np.percentile(arr, 75)) if len(arr) else 0.0,
        "max_dist_m": float(arr.max()) if len(arr) else 0.0,
        "mean_dist_m": float(arr.mean()) if len(arr) else 0.0,
        "los_ratio": float(los_count) / max(1, len(md_positions)),
    }
    print(stats)
    if args.sample and len(md_positions):
        sample = md_positions[: min(args.sample, len(md_positions))]
        print({"sample_md_positions": sample})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
