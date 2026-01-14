"""Deterministic world generation: terrain, city, lakes, obstacles, waypoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
import numpy as np

from framework.config import CityConfig, LakeConfig, TerrainConfig, UncertainObstacleConfig, WorldConfig
from framework.seeding import make_rng
from framework.env.world import Building, Lake, Obstacle, Terrain, WaypointGraph, World


class ValueNoise2D:
    def __init__(self, rng: np.random.Generator, grid_size: int, amplitude: float) -> None:
        self.grid_size = grid_size
        self.amplitude = amplitude
        self.grid = rng.random((grid_size + 1, grid_size + 1)) * 2.0 - 1.0

    def value(self, x: float, y: float) -> float:
        gx = x * self.grid_size
        gy = y * self.grid_size
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        fx = gx - ix
        fy = gy - iy
        ix = max(0, min(self.grid_size - 1, ix))
        iy = max(0, min(self.grid_size - 1, iy))
        v00 = self.grid[ix, iy]
        v10 = self.grid[ix + 1, iy]
        v01 = self.grid[ix, iy + 1]
        v11 = self.grid[ix + 1, iy + 1]
        vx0 = v00 * (1 - fx) + v10 * fx
        vx1 = v01 * (1 - fx) + v11 * fx
        return (vx0 * (1 - fy) + vx1 * fy) * self.amplitude


@dataclass
class TerrainBuilder:
    cfg: TerrainConfig
    rng: np.random.Generator

    def build(self) -> Terrain:
        octaves = max(1, self.cfg.octaves)
        noises: List[ValueNoise2D] = []
        amp = 1.0
        base_grid = 4
        for i in range(octaves):
            grid = int(base_grid * (self.cfg.lacunarity ** i))
            grid = max(1, grid)
            noises.append(ValueNoise2D(self.rng, grid, amp))
            amp *= self.cfg.gain

        def height_fn(x: float, y: float) -> float:
            # Normalize to [0,1] for noise lookup
            nx = (x + 0.5 * self.cfg.size_m) / self.cfg.size_m
            ny = (y + 0.5 * self.cfg.size_m) / self.cfg.size_m
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            val = 0.0
            for n in noises:
                val += n.value(nx, ny)
            # Ridge + valley shaping
            ridge = 1.0 - abs(val)
            valley = -abs(val)
            val = (1.0 - self.cfg.ridge_strength) * val + self.cfg.ridge_strength * ridge
            val = (1.0 - self.cfg.valley_strength) * val + self.cfg.valley_strength * valley
            # Radial shaping (mountain ring)
            if self.cfg.radial_mountain:
                cx, cy = 0.0, 0.0
                r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                r_norm = r / (0.5 * self.cfg.size_m)
                ring = max(0.0, min(1.0, r_norm))
                val *= (self.cfg.radial_base + self.cfg.radial_gain * ring * self.cfg.radial_scale)
            # Scale and clamp
            height = self.cfg.base_height_m + val * self.cfg.max_height_m
            return max(self.cfg.base_height_m, height)

        return Terrain(height_fn=height_fn, clearance_m=self.cfg.clearance_m)


def generate_lakes(cfg: LakeConfig, rng: np.random.Generator, world_size: float) -> List[Lake]:
    if not cfg.enabled:
        return []
    lakes: List[Lake] = []
    for _ in range(cfg.count):
        cx = (rng.random() - 0.5) * world_size * cfg.placement_scale
        cy = (rng.random() - 0.5) * world_size * cfg.placement_scale
        rx = rng.uniform(cfg.min_radius_m, cfg.max_radius_m)
        ry = rng.uniform(cfg.min_radius_m, cfg.max_radius_m)
        rot = rng.uniform(0, math.pi)
        lakes.append(Lake(cx=cx, cy=cy, rx=rx, ry=ry, rot=rot, buffer_m=cfg.buffer_m))
    return lakes


def is_in_lake(x: float, y: float, lake: Lake) -> bool:
    dx = x - lake.cx
    dy = y - lake.cy
    cos_r = math.cos(-lake.rot)
    sin_r = math.sin(-lake.rot)
    nx = dx * cos_r - dy * sin_r
    ny = dx * sin_r + dy * cos_r
    return (nx * nx) / (lake.rx ** 2) + (ny * ny) / (lake.ry ** 2) <= 1.0


def generate_city(cfg: CityConfig, rng: np.random.Generator, terrain: Terrain, lakes: List[Lake]) -> List[Building]:
    if not cfg.enabled:
        return []
    buildings: List[Building] = []
    footprints: List[Tuple[float, float, float, float]] = []
    attempts = 0
    max_attempts = cfg.max_buildings * cfg.max_attempts_factor
    while len(buildings) < cfg.max_buildings and attempts < max_attempts:
        attempts += 1
        x = (rng.random() - 0.5) * cfg.radius_m * 2.0
        y = (rng.random() - 0.5) * cfg.radius_m * 2.0
        r = math.sqrt(x * x + y * y)
        if r > cfg.radius_m:
            continue
        # Park exclusion
        if r < cfg.park_radius_m:
            continue
        # Road exclusion (grid)
        if abs(x) % cfg.road_grid_m < cfg.road_width_m or abs(y) % cfg.road_grid_m < cfg.road_width_m:
            continue
        # Lake exclusion
        if any(is_in_lake(x, y, lake) for lake in lakes):
            continue
        # Density filter
        if rng.random() > cfg.building_density:
            continue
        w = rng.uniform(cfg.footprint_min_m, cfg.footprint_max_m)
        d = rng.uniform(cfg.footprint_min_m, cfg.footprint_max_m)
        # Avoid overlapping footprints
        overlap = False
        for fx, fy, fw, fd in footprints:
            if abs(x - fx) < (w + fw) / 2 and abs(y - fy) < (d + fd) / 2:
                overlap = True
                break
        if overlap:
            continue
        height = rng.uniform(cfg.min_height_m, cfg.max_height_m)
        z0 = terrain.height(x, y)
        buildings.append(Building(x=x, y=y, width=w, depth=d, height=height, base_z=z0))
        footprints.append((x, y, w, d))
    return buildings


def generate_obstacles(cfg: UncertainObstacleConfig, rng: np.random.Generator, world_size: float, terrain: Terrain) -> List[Obstacle]:
    if not cfg.enabled:
        return []
    obs: List[Obstacle] = []
    for _ in range(cfg.count):
        x = (rng.random() - 0.5) * world_size * cfg.placement_scale
        y = (rng.random() - 0.5) * world_size * cfg.placement_scale
        r = rng.uniform(cfg.min_radius_m, cfg.max_radius_m)
        z = terrain.height(x, y) + rng.uniform(cfg.min_altitude_offset_m, cfg.max_altitude_offset_m)
        obs.append(Obstacle(x=x, y=y, z=z, radius=r))
    return obs


def build_waypoint_graph(world: World, rng: np.random.Generator) -> WaypointGraph:
    nodes: List[Tuple[float, float, float]] = []
    attempts = 0
    max_attempts = world.cfg.waypoint_count * world.cfg.waypoint_attempts_factor
    while len(nodes) < world.cfg.waypoint_count and attempts < max_attempts:
        attempts += 1
        x = (rng.random() - 0.5) * world.cfg.size_m
        y = (rng.random() - 0.5) * world.cfg.size_m
        z = rng.uniform(world.cfg.waypoint_altitude_min_m, world.cfg.waypoint_altitude_max_m)
        if world.is_free((x, y, z)):
            nodes.append((x, y, z))
    graph = WaypointGraph(nodes=nodes)
    graph.build_edges(world, radius=world.cfg.connect_radius_m, step=world.cfg.connect_step_m)
    return graph


def generate_world(cfg: WorldConfig) -> World:
    rng = make_rng(cfg.seed)
    terrain_cfg = cfg.terrain
    terrain_cfg.size_m = cfg.size_m
    tb = TerrainBuilder(cfg=terrain_cfg, rng=rng)
    terrain = tb.build()

    lakes = generate_lakes(cfg.lakes, rng, cfg.size_m)
    buildings = generate_city(cfg.city, rng, terrain, lakes)
    obstacles = generate_obstacles(cfg.obstacles, rng, cfg.size_m, terrain)

    world = World(cfg=cfg, terrain=terrain, buildings=buildings, lakes=lakes, obstacles=obstacles)
    world.waypoints = build_waypoint_graph(world, rng)
    return world
