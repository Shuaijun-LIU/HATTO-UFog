"""World definitions and collision checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple

import math

from framework.config import WorldConfig


@dataclass
class Terrain:
    height_fn: Callable[[float, float], float]
    clearance_m: float = 3.0

    def height(self, x: float, y: float) -> float:
        return self.height_fn(x, y)

    def is_free(self, point: Tuple[float, float, float]) -> bool:
        x, y, z = point
        return z > self.height(x, y) + self.clearance_m


@dataclass
class Building:
    x: float
    y: float
    width: float
    depth: float
    height: float
    base_z: float

    def contains(self, point: Tuple[float, float, float]) -> bool:
        px, py, pz = point
        if abs(px - self.x) > self.width / 2:
            return False
        if abs(py - self.y) > self.depth / 2:
            return False
        return self.base_z <= pz <= self.base_z + self.height


@dataclass
class Lake:
    cx: float
    cy: float
    rx: float
    ry: float
    rot: float
    buffer_m: float


@dataclass
class Obstacle:
    x: float
    y: float
    z: float
    radius: float

    def contains(self, point: Tuple[float, float, float]) -> bool:
        px, py, pz = point
        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        return dx * dx + dy * dy + dz * dz <= self.radius * self.radius


@dataclass
class WaypointGraph:
    nodes: List[Tuple[float, float, float]]
    edges: List[List[int]] = field(default_factory=list)

    def build_edges(self, world: "World", radius: float, step: float) -> None:
        n = len(self.nodes)
        self.edges = [[] for _ in range(n)]
        for i in range(n):
            xi, yi, zi = self.nodes[i]
            for j in range(i + 1, n):
                xj, yj, zj = self.nodes[j]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > radius:
                    continue
                if world.segment_is_free((xi, yi, zi), (xj, yj, zj), step=step):
                    self.edges[i].append(j)
                    self.edges[j].append(i)


@dataclass
class World:
    cfg: WorldConfig
    terrain: Terrain
    buildings: List[Building]
    lakes: List[Lake]
    obstacles: List[Obstacle]
    waypoints: WaypointGraph | None = None

    def is_free(self, point: Tuple[float, float, float]) -> bool:
        if not self.terrain.is_free(point):
            return False
        for b in self.buildings:
            if b.contains(point):
                return False
        for o in self.obstacles:
            if o.contains(point):
                return False
        return True

    def segment_is_free(self, p0: Tuple[float, float, float], p1: Tuple[float, float, float], step: float = 5.0) -> bool:
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist == 0:
            return self.is_free(p0)
        steps = max(1, int(dist / step))
        for i in range(steps + 1):
            t = i / steps
            x = x0 + dx * t
            y = y0 + dy * t
            z = z0 + dz * t
            if not self.is_free((x, y, z)):
                return False
        return True
