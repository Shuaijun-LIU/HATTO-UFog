"""Analyze MD service-point reachability via waypoint graph."""
from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufog_network.config import load_config
from ufog_network.env.generators import generate_world
from ufog_network.env.tasks import generate_md_positions
from ufog_network.seeding import make_rng


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config yaml/json")
    parser.add_argument("--sample", type=int, default=0, help="Sample count to print (0 disables)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world = generate_world(cfg.world)
    md_positions, md_service_positions = generate_md_positions(world, cfg.tasks, cfg.sim.seed)

    if not world.waypoints or not world.waypoints.nodes:
        print({"error": "no_waypoints", "md_count": len(md_positions)})
        return 1

    # initial UAV position (same logic as Simulator)
    z0 = world.terrain.height(0.0, 0.0) + world.terrain.clearance_m + cfg.sim.initial_altitude_margin_m
    init_pos = (0.0, 0.0, z0)
    if not world.is_free(init_pos):
        # fallback: search for a free point, same as simulator
        rng = make_rng(cfg.sim.seed)
        for _ in range(cfg.sim.init_pos_attempts):
            x = (rng.random() - 0.5) * world.cfg.size_m
            y = (rng.random() - 0.5) * world.cfg.size_m
            z = world.terrain.height(x, y) + world.terrain.clearance_m + cfg.sim.initial_altitude_margin_m
            if world.is_free((x, y, z)):
                init_pos = (x, y, z)
                break

    nodes = world.waypoints.nodes
    edges = world.waypoints.edges

    def nearest_reachable_node(point: tuple[float, float, float]) -> int | None:
        candidates = sorted(
            range(len(nodes)),
            key=lambda i: (nodes[i][0] - point[0]) ** 2 + (nodes[i][1] - point[1]) ** 2 + (nodes[i][2] - point[2]) ** 2,
        )
        for idx in candidates:
            if world.segment_is_free(point, nodes[idx], step=world.cfg.connect_step_m):
                return idx
        return None

    start = nearest_reachable_node(init_pos)
    if start is None:
        print({"error": "no_reachable_start_node", "md_count": len(md_positions)})
        return 1

    # connected component from start
    comp = set([start])
    q = deque([start])
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v not in comp:
                comp.add(v)
                q.append(v)

    max_dist = cfg.tasks.md_service_waypoint_max_dist_m
    if max_dist <= 0.0:
        max_dist = world.cfg.connect_radius_m
    require_los = cfg.tasks.md_service_waypoint_require_los

    unreachable = []
    stats = {
        "md_count": len(md_positions),
        "service_point_free": 0,
        "has_waypoint_connection": 0,
        "graph_reachable": 0,
    }

    for idx, sp in enumerate(md_service_positions):
        x, y, _z = md_positions[idx]
        stats["service_point_free"] += 1
        # find candidate waypoints near service point
        candidates = []
        for i, n in enumerate(nodes):
            dx = n[0] - sp[0]
            dy = n[1] - sp[1]
            dz = n[2] - sp[2]
            if dx * dx + dy * dy + dz * dz <= max_dist * max_dist:
                if not require_los or world.segment_is_free(sp, n, step=world.cfg.connect_step_m):
                    candidates.append(i)
        if not candidates:
            unreachable.append({"md_idx": idx, "reason": "no_waypoint_connection", "pos": (x, y)})
            continue
        stats["has_waypoint_connection"] += 1
        if not any(i in comp for i in candidates):
            unreachable.append({"md_idx": idx, "reason": "no_graph_path", "pos": (x, y)})
            continue
        stats["graph_reachable"] += 1

    report = {
        "md_count": stats["md_count"],
        "service_point_free_ratio": stats["service_point_free"] / max(1, stats["md_count"]),
        "waypoint_connection_ratio": stats["has_waypoint_connection"] / max(1, stats["md_count"]),
        "graph_reachable_ratio": stats["graph_reachable"] / max(1, stats["md_count"]),
        "unreachable_count": len(unreachable),
        "unreachable": unreachable[: args.sample] if args.sample else [],
    }
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
