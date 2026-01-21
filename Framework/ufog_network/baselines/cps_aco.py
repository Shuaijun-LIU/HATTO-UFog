"""CPS-ACO baseline with chaotic mapping, polarized pheromone, and SA screening."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from ufog_network.baselines.base import Action, Baseline
from ufog_network.seeding import make_rng


class CPSACOBaseline(Baseline):
    name = "cps_aco"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def reset(self, world, seed: int) -> None:
        super().reset(world, seed)
        self.rng = make_rng(seed)
        self.graph = world.waypoints if world.waypoints and world.waypoints.nodes else None
        if self.graph and self.graph.nodes:
            n = len(self.graph.nodes)
            self.tau = np.ones((n, n), dtype=np.float64)
            self.chaos = np.ones((n, n), dtype=np.float64) * 0.5
            # Precompute edge costs for fast neighbor scoring
            self.edge_cost_cache = {}
            self.edge_len_cache = {}
            edges = self.graph.edges
            for i, nbrs in enumerate(edges):
                for j in nbrs:
                    self.edge_cost_cache[(i, j)] = self._edge_cost(i, j)
                    xi, yi, zi = self.graph.nodes[i]
                    xj, yj, zj = self.graph.nodes[j]
                    self.edge_len_cache[(i, j)] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
        else:
            self.tau = None
            self.chaos = None
            self.edge_cost_cache = {}
            self.edge_len_cache = {}
        self.current_path: List[Tuple[float, float, float]] = []
        self.last_target_idx: int | None = None

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

    def _nearest_reachable_node(self, point: Tuple[float, float, float]) -> int:
        nodes = self.graph.nodes
        candidates = sorted(
            range(len(nodes)),
            key=lambda i: (nodes[i][0] - point[0]) ** 2 + (nodes[i][1] - point[1]) ** 2 + (nodes[i][2] - point[2]) ** 2,
        )
        limit = int(self.params.get("reachable_candidates", 0))
        if limit > 0:
            candidates = candidates[: min(limit, len(candidates))]
        for idx in candidates:
            if self.world.segment_is_free_cached(point, nodes[idx], step=self.world.cfg.connect_step_m):
                return idx
        return self._nearest_node(point)

    def _edge_cost(self, i: int, j: int) -> float:
        xi, yi, zi = self.graph.nodes[i]
        xj, yj, zj = self.graph.nodes[j]
        dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
        # Obstacle threat cost: inverse distance to nearest obstacle
        threat = 0.0
        threat_buffer = float(self.params.get("threat_buffer_m", 5.0))
        threat_near = float(self.params.get("threat_near_penalty", 10.0))
        threat_far_scale = float(self.params.get("threat_far_scale", 1.0))
        for o in self.world.obstacles:
            dx = xj - o.x
            dy = yj - o.y
            dz = zj - o.z
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < o.radius + threat_buffer:
                threat += threat_near
            else:
                threat += threat_far_scale / (d + 1e-6)
        # Constraint cost proxy: large altitude jumps
        alt_jump = float(self.params.get("alt_jump_threshold_m", 20.0))
        constraint = max(0.0, abs(zj - zi) - alt_jump)
        w_len = float(self.params.get("lambda_len", 1.0))
        w_threat = float(self.params.get("lambda_threat", 1.0))
        w_const = float(self.params.get("lambda_const", 1.0))
        return w_len * dist + w_threat * threat + w_const * constraint

    def _build_path(self, start_idx: int, goal_idx: int) -> List[int]:
        n = len(self.graph.nodes)
        alpha = float(self.params.get("alpha", 1.0))
        beta = float(self.params.get("beta", 2.0))
        delta = float(self.params.get("delta", 0.2))
        epsilon = float(self.params.get("epsilon", 0.2))
        mu = float(self.params.get("mu", 4.0))
        ants = int(self.params.get("ants", 30))
        iterations = int(self.params.get("iterations", 40))
        temp = float(self.params.get("sa_temp", 1.0))
        temp_min = float(self.params.get("sa_temp_min", 0.05))
        temp_decay = float(self.params.get("sa_temp_decay", 0.95))

        nodes = self.graph.nodes
        edges = self.graph.edges

        def dist(i: int, j: int) -> float:
            cached = self.edge_len_cache.get((i, j))
            if cached is not None:
                return cached
            xi, yi, zi = nodes[i]
            xj, yj, zj = nodes[j]
            return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

        best_path = None
        best_cost = 1e18

        for _ in range(iterations):
            paths = []
            costs = []
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
                    # Chaotic mapping update for current row
                    self.chaos[current, nbrs] = mu * self.chaos[current, nbrs] * (1.0 - self.chaos[current, nbrs])
                    self.tau[current, nbrs] = (1.0 - delta) * self.tau[current, nbrs] + epsilon * self.chaos[current, nbrs]

                    probs = []
                    for k in nbrs:
                        tau = self.tau[current, k] ** alpha
                        eta = (1.0 / (dist(current, k) + 1e-6)) ** beta
                        probs.append(tau * eta)
                    probs = np.array(probs, dtype=np.float64)
                    denom = probs.sum()
                    if denom <= 1e-12:
                        break
                    probs = probs / denom
                    # SA screening: sample candidate, then accept based on cost
                    cand_idx = self.rng.choice(len(nbrs), p=probs)
                    cand = nbrs[cand_idx]
                    # Compare with best local choice
                    local_best = min(nbrs, key=lambda k: self.edge_cost_cache.get((current, k), self._edge_cost(current, k)))
                    cand_cost = self.edge_cost_cache.get((current, cand), self._edge_cost(current, cand))
                    best_cost = self.edge_cost_cache.get((current, local_best), self._edge_cost(current, local_best))
                    dE = cand_cost - best_cost
                    if dE > 0 and self.rng.random() > math.exp(-dE / max(temp, 1e-6)):
                        cand = local_best
                    current = cand
                    path.append(current)
                    visited.add(current)
                if path[-1] != goal_idx:
                    continue
                cost = sum(self.edge_cost_cache.get((path[i], path[i + 1]), self._edge_cost(path[i], path[i + 1])) for i in range(len(path) - 1))
                paths.append(path)
                costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # Polarized pheromone recording
            if paths:
                order = np.argsort(costs)
                top_k = max(1, int(0.2 * len(paths)))
                bottom_k = max(1, int(0.2 * len(paths)))
                for rank, idx in enumerate(order[:top_k]):
                    path = paths[idx]
                    weight = (top_k - rank) / max(1, top_k)
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        self.tau[a, b] += weight / (costs[idx] + 1e-6)
                        self.tau[b, a] += weight / (costs[idx] + 1e-6)
                for idx in order[-bottom_k:]:
                    path = paths[idx]
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        self.tau[a, b] *= 0.5
                        self.tau[b, a] *= 0.5

            temp = max(temp * temp_decay, temp_min)

        return best_path or [start_idx, goal_idx]

    def act(self, state: Dict[str, Any]) -> Action:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return Action(target=pos, info={"status": "done"})
        target = targets[target_idx]

        if not self.graph or not self.graph.nodes:
            return Action(target=target, info={"status": "ok", "target_idx": target_idx})

        if self.last_target_idx is None:
            self.last_target_idx = target_idx
        elif target_idx != self.last_target_idx:
            # Invalidate cached path when the active target changes.
            self.current_path = []
            self.last_target_idx = target_idx

        if not self.current_path:
            start = self._nearest_reachable_node(pos)
            goal = self._nearest_node(target)
            node_path = self._build_path(start, goal)
            self.current_path = [self.graph.nodes[i] for i in node_path]
            if not self.current_path:
                self.current_path = [target]
            else:
                last = self.current_path[-1]
                if (last[0] - target[0]) ** 2 + (last[1] - target[1]) ** 2 + (last[2] - target[2]) ** 2 > 1e-6:
                    self.current_path.append(target)

        reach = float(self.params.get("waypoint_reach_m", 8.0))
        while self.current_path:
            next_wp = self.current_path[0]
            if math.sqrt((pos[0] - next_wp[0]) ** 2 + (pos[1] - next_wp[1]) ** 2 + (pos[2] - next_wp[2]) ** 2) <= reach:
                self.current_path.pop(0)
                continue
            break
        if not self.current_path:
            next_wp = target
        else:
            next_wp = self.current_path[0]
        return Action(target=next_wp, info={"status": "ok", "target_idx": target_idx})

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return {"status": "done", "path": []}
        target = targets[target_idx]
        if not self.graph or not self.graph.nodes:
            return {"status": "ok", "path": [target], "target_idx": target_idx}
        start = self._nearest_reachable_node(pos)
        goal = self._nearest_node(target)
        node_path = self._build_path(start, goal)
        path = [self.graph.nodes[i] for i in node_path]
        if not path:
            path = [target]
        else:
            last = path[-1]
            if (last[0] - target[0]) ** 2 + (last[1] - target[1]) ** 2 + (last[2] - target[2]) ** 2 > 1e-6:
                path.append(target)
        return {"status": "ok", "path": path, "target_idx": target_idx}
