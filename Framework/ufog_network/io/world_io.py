"""World export/import utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from dataclasses import asdict

from ufog_network.config import Config
from ufog_network.env.generators import TerrainBuilder
from ufog_network.env.world import Building, Lake, Obstacle, WaypointGraph, World
from ufog_network.seeding import make_rng


def _export_heightmap(world: World) -> Dict[str, Any] | None:
    step = float(world.cfg.heightmap_step_m)
    if step <= 0:
        return None
    extent = float(world.cfg.heightmap_extent_m or world.cfg.size_m)
    half = extent / 2.0
    x_vals = []
    y_vals = []
    x = -half
    while x <= half + 1e-6:
        x_vals.append(float(x))
        x += step
    y = -half
    while y <= half + 1e-6:
        y_vals.append(float(y))
        y += step
    heights = []
    for yy in y_vals:
        row = []
        for xx in x_vals:
            row.append(float(world.terrain.height(xx, yy)))
        heights.append(row)
    return {
        "step_m": step,
        "extent_m": extent,
        "origin": [-half, -half],
        "heights": heights,
    }


def export_world(world: World, path: str) -> None:
    payload: Dict[str, Any] = {
        "config": asdict(world.cfg),
        "buildings": [b.__dict__ for b in world.buildings],
        "lakes": [l.__dict__ for l in world.lakes],
        "obstacles": [o.__dict__ for o in world.obstacles],
        "waypoints": {
            "nodes": world.waypoints.nodes if world.waypoints else [],
            "edges": world.waypoints.edges if world.waypoints else [],
        },
    }
    heightmap = _export_heightmap(world)
    if heightmap is not None:
        payload["heightmap"] = heightmap
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def import_world(path: str) -> World:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"World file not found: {path}")
    payload = json.loads(p.read_text())
    cfg = Config.from_dict({"world": payload["config"]}).world
    rng = make_rng(cfg.seed)
    terrain_cfg = cfg.terrain
    terrain_cfg.size_m = cfg.size_m
    terrain = TerrainBuilder(cfg=terrain_cfg, rng=rng).build()
    buildings = [Building(**b) for b in payload.get("buildings", [])]
    lakes = [Lake(**l) for l in payload.get("lakes", [])]
    obstacles = [Obstacle(**o) for o in payload.get("obstacles", [])]
    world = World(cfg=cfg, terrain=terrain, buildings=buildings, lakes=lakes, obstacles=obstacles)
    wp = payload.get("waypoints", {})
    if wp:
        graph = WaypointGraph(nodes=wp.get("nodes", []), edges=wp.get("edges", []))
        world.waypoints = graph
    return world
