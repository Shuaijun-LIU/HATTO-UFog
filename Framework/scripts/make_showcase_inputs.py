"""Generate deterministic showcase inputs (world + safe 3D route).

Outputs:
- showcase_world.json: environment geometry (terrain heightmap + buildings + lakes + obstacles)
- showcase_trajectory.json: obstacle-aware route for playback (path points + speed)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufog_network.config import load_config
from ufog_network.env.generators import generate_world
from ufog_network.io.world_io import export_world
from ufog_network.seeding import make_rng
from ufog_network.utils import shortest_path


def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _nearest_reachable_node(
    world,
    point: Tuple[float, float, float],
    max_candidates: int = 80,
) -> int:
    nodes = world.waypoints.nodes
    order = sorted(range(len(nodes)), key=lambda i: _dist2(nodes[i], point))
    step = float(world.cfg.connect_step_m)
    for idx in order[: min(max_candidates, len(order))]:
        if world.segment_is_free(point, nodes[idx], step=step):
            return idx
    return int(order[0]) if order else 0


def _pick_free_start(world, rng: np.random.Generator, attempts: int = 500) -> Tuple[float, float, float]:
    for _ in range(attempts):
        x = (rng.random() - 0.5) * 0.2 * world.cfg.size_m
        y = (rng.random() - 0.5) * 0.2 * world.cfg.size_m
        z = world.terrain.height(x, y) + world.terrain.clearance_m + 55.0
        p = (float(x), float(y), float(z))
        if world.is_free(p):
            return p
    # Fallback: origin at a conservative altitude.
    z0 = world.terrain.height(0.0, 0.0) + world.terrain.clearance_m + 60.0
    return (0.0, 0.0, float(z0))


def _sample_region_nodes(nodes: List[Tuple[float, float, float]], predicate) -> List[int]:
    out = []
    for i, (x, y, z) in enumerate(nodes):
        if predicate(float(x), float(y), float(z)):
            out.append(i)
    return out


def _choose_showcase_route(world, seed: int) -> List[int]:
    if not world.waypoints or not world.waypoints.nodes or not world.waypoints.edges:
        raise RuntimeError("Waypoint graph is missing; cannot build a robust obstacle-avoiding route.")

    rng = make_rng(seed + 999)
    nodes = world.waypoints.nodes

    start_pos = _pick_free_start(world, rng)
    start_idx = _nearest_reachable_node(world, start_pos)

    city_r = float(getattr(world.cfg.city, "radius_m", 350.0))
    city_nodes = _sample_region_nodes(
        nodes,
        lambda x, y, z: math.sqrt(x * x + y * y) < 0.75 * city_r and 40.0 < z < 180.0,
    )
    mountain_nodes = _sample_region_nodes(
        nodes,
        lambda x, y, z: math.sqrt(x * x + y * y) > 0.42 * world.cfg.size_m and z > 140.0,
    )

    lake_nodes: List[int] = []
    for lake in world.lakes:
        cx, cy = float(lake.cx), float(lake.cy)
        r = max(float(lake.rx), float(lake.ry)) + 40.0
        r2 = r * r
        lake_nodes.extend(
            _sample_region_nodes(
                nodes,
                lambda x, y, z, cx=cx, cy=cy, r2=r2: ((x - cx) ** 2 + (y - cy) ** 2) < r2 and z < 200.0,
            )
        )

    if not city_nodes or not mountain_nodes:
        raise RuntimeError("Not enough candidate nodes to build a showcase route; increase waypoint_count or relax filters.")

    # Try multiple random anchor selections until routing succeeds.
    for _ in range(300):
        city_idx = int(rng.choice(city_nodes))
        mountain_idx = int(rng.choice(mountain_nodes))
        lake_idx = int(rng.choice(lake_nodes)) if lake_nodes else int(rng.integers(0, len(nodes)))

        anchors = [start_idx, city_idx, mountain_idx, lake_idx, start_idx]
        full: List[int] = []
        ok = True
        for a, b in zip(anchors[:-1], anchors[1:]):
            sp = shortest_path(nodes, world.waypoints.edges, int(a), int(b))
            if not sp:
                ok = False
                break
            if not full:
                full.extend(sp)
            else:
                full.extend(sp[1:])
        if ok and len(full) >= 6:
            return full

    raise RuntimeError("Failed to build a valid showcase route after many attempts.")


def _densify_path(points: List[Tuple[float, float, float]], step_m: float) -> List[Tuple[float, float, float]]:
    if len(points) < 2:
        return points[:]
    out: List[Tuple[float, float, float]] = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        ax, ay, az = a
        bx, by, bz = b
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = max(1, int(math.ceil(dist / max(1e-6, step_m))))
        for i in range(1, n + 1):
            t = i / n
            out.append((ax + dx * t, ay + dy * t, az + dz * t))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/showcase.yaml", help="Config yaml for world generation")
    parser.add_argument("--output-dir", type=str, default="showcase/inputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override showcase seed (optional)")
    parser.add_argument("--speed-m-s", type=float, default=14.0, help="Playback speed (m/s)")
    parser.add_argument("--densify-step-m", type=float, default=6.0, help="Densify route points every N meters")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.world.seed = int(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = generate_world(cfg.world)

    world_path = out_dir / "showcase_world.json"
    export_world(world, str(world_path))

    # For showcase playback we do not need the full waypoint graph (it can be large).
    payload = json.loads(world_path.read_text())
    if "waypoints" in payload:
        payload["waypoints"]["edges"] = []
    # Minify to keep the heightmap reasonably sized for open-source distribution.
    world_path.write_text(json.dumps(payload, separators=(",", ":")))

    route_node_path = _choose_showcase_route(world, seed=cfg.world.seed)
    route = [tuple(map(float, world.waypoints.nodes[i])) for i in route_node_path]
    route = _densify_path(route, step_m=float(args.densify_step_m))

    traj = {
        "meta": {
            "seed": int(cfg.world.seed),
            "units": "m",
            "coord": "sim(x,y,z) with z up",
            "speed_m_s": float(args.speed_m_s),
            "notes": "Route is built on the waypoint graph using collision-free edges (showcase playback).",
        },
        "path": [[float(x), float(y), float(z)] for (x, y, z) in route],
    }
    traj_path = out_dir / "showcase_trajectory.json"
    traj_path.write_text(json.dumps(traj, indent=2))

    print(f"Wrote: {world_path}")
    print(f"Wrote: {traj_path}")


if __name__ == "__main__":
    main()
