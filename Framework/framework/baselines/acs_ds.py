"""ACS-DS trajectory planner with safety values and decoupling."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from framework.baselines.base import Action, Baseline
from framework.seeding import make_rng


class ACSDSBaseline(Baseline):
    name = "acs_ds"

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

    def _safety_value(self, point: Tuple[float, float, float]) -> float:
        # Approximate safety with random sampling in a local radius.
        samples = int(self.params.get("safety_samples", 30))
        radius = float(self.params.get("safety_radius_m", 120.0))
        if samples <= 0:
            return 1.0
        free = 0
        for _ in range(samples):
            # Uniform sample in a sphere
            u = self.rng.random()
            v = self.rng.random()
            theta = 2 * math.pi * u
            phi = math.acos(2 * v - 1)
            r = radius * (self.rng.random() ** (1 / 3))
            dx = r * math.sin(phi) * math.cos(theta)
            dy = r * math.sin(phi) * math.sin(theta)
            dz = r * math.cos(phi)
            p = (point[0] + dx, point[1] + dy, point[2] + dz)
            if self.world.is_free(p):
                free += 1
        return free / max(1, samples)

    def _build_path(self, start_idx: int, goal_idx: int) -> Tuple[List[int], int]:
        n = len(self.graph.nodes)
        if self.pheromone is None:
            self.pheromone = np.ones((n, n), dtype=np.float64)
        alpha = float(self.params.get("alpha", 1.0))
        beta = float(self.params.get("beta", 2.0))
        rho = float(self.params.get("rho", 0.2))
        ants = int(self.params.get("ants", 30))
        iterations = int(self.params.get("iterations", 40))
        safety_weight = float(self.params.get("safety_weight", 1.0))
        backtrack_steps = int(self.params.get("backtrack_steps", 4))
        safety_drop_ratio = float(self.params.get("safety_drop_ratio", 0.5))
        max_stale_steps = int(self.params.get("max_stale_steps", 25))

        best_path = None
        best_len = 1e18
        best_backtracks = 0

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
                prev_kappa = None
                last_backtrack = 0
                backtracks = 0
                for step in range(n):
                    if current == goal_idx:
                        break
                    nbrs = edges[current]
                    if not nbrs:
                        break
                    # Compute safety values per neighbor
                    kappas = [self._safety_value(nodes[k]) for k in nbrs]
                    probs = []
                    for idx, k in enumerate(nbrs):
                        tau = self.pheromone[current, k] ** alpha
                        eta = (1.0 / (dist(current, k) + 1e-6)) ** beta
                        sigma = eta * (1.0 + safety_weight * kappas[idx])
                        probs.append(tau * sigma)
                    probs = np.array(probs, dtype=np.float64)
                    probs = probs / probs.sum()
                    next_idx = self.rng.choice(len(nbrs), p=probs)
                    nxt = nbrs[next_idx]
                    kappa = kappas[next_idx]

                    # Decoupling/backtracking triggers
                    loop = nxt in visited
                    safety_drop = prev_kappa is not None and kappa < prev_kappa * safety_drop_ratio
                    stale = (step - last_backtrack) > max_stale_steps
                    if loop or safety_drop or stale:
                        backtracks += 1
                        if len(path) > 1:
                            cut = min(backtrack_steps, len(path) - 1)
                            path = path[:-cut]
                            current = path[-1]
                            visited = set(path)
                            prev_kappa = None
                            last_backtrack = step
                            continue
                    current = nxt
                    path.append(current)
                    visited.add(current)
                    prev_kappa = kappa

                if path[-1] != goal_idx:
                    continue
                length = sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))
                if length < best_len:
                    best_len = length
                    best_path = path
                    best_backtracks = backtracks

            # pheromone evaporation + best update
            self.pheromone *= (1.0 - rho)
            if best_path:
                for i in range(len(best_path) - 1):
                    a = best_path[i]
                    b = best_path[i + 1]
                    self.pheromone[a, b] += 1.0 / (best_len + 1e-6)
                    self.pheromone[b, a] += 1.0 / (best_len + 1e-6)

        return best_path or [start_idx, goal_idx], best_backtracks

    def act(self, state: Dict[str, Any]) -> Action:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return Action(target=pos, info={"status": "done"})
        target = targets[target_idx]

        if not self.graph or not self.graph.nodes:
            return Action(target=target, info={"status": "ok", "target_idx": target_idx})

        if not self.current_path:
            start = self._nearest_node(pos)
            goal = self._nearest_node(target)
            node_path, backtracks = self._build_path(start, goal)
            self.current_path = [self.graph.nodes[i] for i in node_path]
            info = {"status": "ok", "target_idx": target_idx, "backtracks": backtracks}
        else:
            info = {"status": "ok", "target_idx": target_idx}

        next_wp = self.current_path.pop(0)
        return Action(target=next_wp, info=info)

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return {"status": "done", "path": []}
        target = targets[target_idx]
        start = self._nearest_node(pos)
        goal = self._nearest_node(target)
        node_path, backtracks = self._build_path(start, goal)
        return {"status": "ok", "path": [self.graph.nodes[i] for i in node_path], "target_idx": target_idx, "backtracks": backtracks}
