"""Validation helpers for runs and worlds."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ufog_network.env.world import World
from ufog_network.schemas import timeseries_columns


def validate_world(world: World) -> Dict[str, bool]:
    results = {
        "buildings_nonempty": True,
        "waypoints_nonempty": True,
        "terrain_variation": True,
    }
    if world.cfg.city.enabled and len(world.buildings) == 0:
        results["buildings_nonempty"] = False
    if world.waypoints is None or len(world.waypoints.nodes) == 0:
        results["waypoints_nonempty"] = False
    # Sample terrain variation
    offsets = [-0.35, -0.1, 0.1, 0.35]
    heights = []
    for ox in offsets:
        for oy in offsets:
            heights.append(world.terrain.height(world.cfg.size_m * ox, world.cfg.size_m * oy))
    if max(heights) - min(heights) < 1e-3:
        results["terrain_variation"] = False
    return results


def validate_waypoint_edges(world: World, sample: int = 500) -> Dict[str, bool]:
    results = {"edges_collision_free": True}
    if world.waypoints is None or not world.waypoints.edges:
        results["edges_collision_free"] = False
        return results
    edges = world.waypoints.edges
    nodes = world.waypoints.nodes
    checked = 0
    for i, nbrs in enumerate(edges):
        for j in nbrs:
            if checked >= sample:
                break
            if not world.segment_is_free(nodes[i], nodes[j], step=world.cfg.connect_step_m):
                results["edges_collision_free"] = False
                return results
            checked += 1
        if checked >= sample:
            break
    return results


def world_fingerprint(world: World) -> Dict[str, float]:
    # Lightweight fingerprint for determinism checks
    bsum = 0.0
    for b in world.buildings[:200]:
        bsum += b.x + b.y + b.width + b.depth + b.height + b.base_z
    osum = 0.0
    for o in world.obstacles[:200]:
        osum += o.x + o.y + o.z + o.radius
    return {
        "buildings": float(len(world.buildings)),
        "obstacles": float(len(world.obstacles)),
        "bsum": float(round(bsum, 3)),
        "osum": float(round(osum, 3)),
    }


def validate_timeseries_schema(path: str) -> Dict[str, bool]:
    cols = [name for name, _dtype in timeseries_columns()]
    df = pd.read_parquet(path)
    results = {}
    for col in cols:
        results[col] = col in df.columns
    return results


def schema_missing(results: Dict[str, bool]) -> List[str]:
    return [k for k, v in results.items() if not v]
