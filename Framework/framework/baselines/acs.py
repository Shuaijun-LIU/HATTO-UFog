"""ACS baseline trajectory planner on waypoint graph."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from framework.baselines.base import Action, Baseline
from framework.seeding import make_rng


class ACSBaseline(Baseline):
    name = "acs"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def reset(self, world, seed: int) -> None:
        super().reset(world, seed)
        self.rng = make_rng(seed)
        self.graph = world.waypoints
        self.pheromone = None
        self.current_path: List[Tuple[float, float, float]] = []

    def _nearest_node(self, point: Tuple[float, float, float]) -> int:
        nodes = self.graph.nodes
        best_idx = 0
        best_d = 1e18
        for i, n in enumerate(nodes):
            d = (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2 + (n[2] - point[2]) ** 2
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx

    def _build_path(self, start_idx: int, goal_idx: int) -> List[int]:
        n = len(self.graph.nodes)
        if self.pheromone is None:
            self.pheromone = np.ones((n, n), dtype=np.float64)
        alpha = float(self.params.get("alpha", 1.0))
        beta = float(self.params.get("beta", 2.0))
        rho = float(self.params.get("rho", 0.2))
        ants = int(self.params.get("ants", 30))
        iterations = int(self.params.get("iterations", 40))

        best_path = None
        best_len = 1e18

        nodes = self.graph.nodes
        edges = self.graph.edges

        def dist(i: int, j: int) -> float:
            xi, yi, zi = nodes[i]
            xj, yj, zj = nodes[j]
            return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

        for _ in range(iterations):
            for _a in range(ants):
                path = [start_idx]
                visited = set([start_idx])
                current = start_idx
                for _step in range(n):
                    if current == goal_idx:
                        break
                    nbrs = [k for k in edges[current] if k not in visited]
                    if not nbrs:
                        break
                    probs = []
                    for k in nbrs:
                        tau = self.pheromone[current, k] ** alpha
                        eta = (1.0 / (dist(current, k) + 1e-6)) ** beta
                        probs.append(tau * eta)
                    probs = np.array(probs, dtype=np.float64)
                    probs = probs / probs.sum()
                    next_idx = self.rng.choice(len(nbrs), p=probs)
                    current = nbrs[next_idx]
                    path.append(current)
                    visited.add(current)
                if path[-1] != goal_idx:
                    continue
                length = sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))
                if length < best_len:
                    best_len = length
                    best_path = path
            # pheromone evaporation
            self.pheromone *= (1.0 - rho)
            if best_path:
                for i in range(len(best_path) - 1):
                    a = best_path[i]
                    b = best_path[i + 1]
                    self.pheromone[a, b] += 1.0 / (best_len + 1e-6)
                    self.pheromone[b, a] += 1.0 / (best_len + 1e-6)

        return best_path or [start_idx, goal_idx]

    def act(self, state: Dict[str, Any]) -> Action:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return Action(target=pos, info={"status": "done"})
        target = targets[target_idx]

        if not self.current_path:
            start = self._nearest_node(pos)
            goal = self._nearest_node(target)
            node_path = self._build_path(start, goal)
            self.current_path = [self.graph.nodes[i] for i in node_path]

        next_wp = self.current_path.pop(0)
        return Action(target=next_wp, info={"status": "ok", "target_idx": target_idx})

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return {"status": "done", "path": []}
        target = targets[target_idx]
        start = self._nearest_node(pos)
        goal = self._nearest_node(target)
        node_path = self._build_path(start, goal)
        return {"status": "ok", "path": [self.graph.nodes[i] for i in node_path], "target_idx": target_idx}
