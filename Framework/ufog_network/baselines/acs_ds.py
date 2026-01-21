"""ACS-DS trajectory planner with safety values and decoupling."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from ufog_network.baselines.base import Action, Baseline
from ufog_network.seeding import make_rng


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
        self.safety_candidates = {}
        self.segment_cache = {}
        self.dist_cache = {}
        self.candidate_vectors = {}
        self.last_plan_slot = -1
        self.last_target_idx = None
        self.last_goal_point = None
        self.last_path_direct = False
        self.segment_step = float(self._param("segment_step_m", self.world.cfg.connect_step_m))
        if self.graph and self.graph.nodes:
            radius = float(self._param("safety_radius_m", 120.0, aliases=("R_f", "R_f_m")))
            r2 = radius * radius
            nodes = self.graph.nodes
            for i, p in enumerate(nodes):
                candidates = []
                for j, q in enumerate(nodes):
                    if i == j:
                        continue
                    dx = q[0] - p[0]
                    dy = q[1] - p[1]
                    dz = q[2] - p[2]
                    if dx * dx + dy * dy + dz * dz <= r2:
                        candidates.append(j)
                self.safety_candidates[i] = candidates
            # Precompute candidate unit vectors for safety evaluation
            for i, p in enumerate(nodes):
                vecs = []
                for j in self.safety_candidates.get(i, []):
                    q = nodes[j]
                    dx = q[0] - p[0]
                    dy = q[1] - p[1]
                    dz = q[2] - p[2]
                    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if norm < 1e-6:
                        continue
                    vecs.append((j, (dx / norm, dy / norm, dz / norm)))
                self.candidate_vectors[i] = vecs

    def _param(self, key: str, default: float | int, aliases: Tuple[str, ...] = ()) -> float | int:
        if key in self.params:
            return self.params[key]
        for alias in aliases:
            if alias in self.params:
                return self.params[alias]
        return default

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
        limit = int(self._param("reachable_candidates", 0))
        if limit > 0:
            candidates = candidates[: min(limit, len(candidates))]
        for idx in candidates:
            if self.world.segment_is_free_cached(point, nodes[idx], step=self.segment_step):
                return idx
        return self._nearest_node(point)

    def _segment_free(self, a_idx: int, b_idx: int) -> bool:
        key = (a_idx, b_idx)
        if key in self.segment_cache:
            return self.segment_cache[key]
        pa = self.graph.nodes[a_idx]
        pb = self.graph.nodes[b_idx]
        free = self.world.segment_is_free_cached(pa, pb, step=self.segment_step)
        self.segment_cache[key] = free
        return free

    def _dist(self, i: int, j: int) -> float:
        key = (i, j)
        if key in self.dist_cache:
            return self.dist_cache[key]
        xi, yi, zi = self.graph.nodes[i]
        xj, yj, zj = self.graph.nodes[j]
        d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
        self.dist_cache[key] = d
        self.dist_cache[(j, i)] = d
        return d

    def _safety_value(self, current_idx: int, next_idx: int) -> float:
        # Safety value kappa_{mu,nu} based on forward feasible ratio.
        if not self.safety_candidates:
            return 1.0
        nodes = self.graph.nodes
        cur = nodes[current_idx]
        nxt = nodes[next_idx]
        dir_vec = (nxt[0] - cur[0], nxt[1] - cur[1], nxt[2] - cur[2])
        norm = math.sqrt(dir_vec[0] ** 2 + dir_vec[1] ** 2 + dir_vec[2] ** 2)
        if norm < 1e-6:
            return 1.0
        dir_unit = (dir_vec[0] / norm, dir_vec[1] / norm, dir_vec[2] / norm)
        angle_deg = float(self.params.get("safety_angle_deg", 90.0))
        cos_limit = math.cos(math.radians(max(1.0, min(179.0, angle_deg))))
        sample_limit = int(self._param("safety_sample_nodes", 0, aliases=("safety_samples",)))
        candidates = self.candidate_vectors.get(current_idx, [])
        if sample_limit > 0 and len(candidates) > sample_limit:
            idxs = self.rng.choice(len(candidates), size=sample_limit, replace=False)
            candidates = [candidates[i] for i in idxs]
        total = 0
        blocked = 0
        for idx, unit in candidates:
            cosang = unit[0] * dir_unit[0] + unit[1] * dir_unit[1] + unit[2] * dir_unit[2]
            if cosang <= cos_limit:
                continue
            total += 1
            if not self._segment_free(current_idx, idx):
                blocked += 1
        if total == 0:
            return 1.0
        return (total - blocked) / total

    def _build_path(self, start_idx: int, goal_idx: int) -> Tuple[List[int], int]:
        n = len(self.graph.nodes)
        if self.pheromone is None:
            pheromone_init = float(self._param("pheromone_init", 1.0, aliases=("V_P0", "V_p0", "vp0")))
            self.pheromone = np.ones((n, n), dtype=np.float64) * pheromone_init
        alpha = float(self.params.get("alpha", 1.0))
        beta = float(self._param("beta", 2.0, aliases=("eta", "eta_pow")))
        rho = float(self.params.get("rho", 0.2))
        ants = int(self.params.get("ants", 30))
        iterations = int(self.params.get("iterations", 40))
        safety_weight = float(self.params.get("safety_weight", 1.0))
        backtrack_steps = int(self.params.get("backtrack_steps", 4))
        backtrack_distance = float(self._param("backtrack_distance_m", 0.0, aliases=("D_b", "D_b_m")))
        safety_drop_ratio = float(self.params.get("safety_drop_ratio", 0.5))
        max_stale_steps = int(self.params.get("max_stale_steps", 25))
        early_iter_ratio = float(self.params.get("early_iter_ratio", 0.33))
        early_stale_steps = int(self.params.get("early_stale_steps", 25))
        pheromone_init = float(self._param("pheromone_init", 1.0, aliases=("V_P0", "V_p0", "vp0")))
        local_evap = float(self._param("local_evap", 0.0, aliases=("phi", "local_rho")))

        best_path = None
        best_len = 1e18
        best_backtracks = 0

        nodes = self.graph.nodes
        edges = self.graph.edges

        for it in range(iterations):
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
                    kappas = [self._safety_value(current, k) for k in nbrs]
                    probs = []
                    for idx, k in enumerate(nbrs):
                        tau = self.pheromone[current, k]
                        eta = (1.0 / (self._dist(current, k) + 1e-6))
                        sigma = tau + safety_weight * kappas[idx]
                        probs.append((sigma ** alpha) * (eta ** beta))
                    probs = np.array(probs, dtype=np.float64)
                    denom = probs.sum()
                    if denom <= 1e-12:
                        break
                    probs = probs / denom
                    next_idx = self.rng.choice(len(nbrs), p=probs)
                    nxt = nbrs[next_idx]
                    kappa = kappas[next_idx]

                    # Decoupling/backtracking triggers
                    loop = nxt in visited
                    safety_drop = prev_kappa is not None and kappa < prev_kappa * safety_drop_ratio
                    stale = (step - last_backtrack) > max_stale_steps
                    early_stale = it < int(iterations * early_iter_ratio) and (step - last_backtrack) > early_stale_steps
                    if loop or safety_drop or stale:
                        backtracks += 1
                        if len(path) > 1:
                            if backtrack_distance > 0:
                                cut = 0
                                dist_acc = 0.0
                                for i in range(len(path) - 1, 0, -1):
                                    a = path[i - 1]
                                    b = path[i]
                                    dist_acc += dist(a, b)
                                    cut += 1
                                    if dist_acc >= backtrack_distance:
                                        break
                                cut = min(cut, len(path) - 1)
                            else:
                                cut = min(backtrack_steps, len(path) - 1)
                            path = path[:-cut]
                            current = path[-1]
                            visited = set(path)
                            prev_kappa = None
                            last_backtrack = step
                            continue
                    if early_stale:
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
                    if local_evap > 0.0:
                        tau = self.pheromone[path[-2], current]
                        updated = (1.0 - local_evap) * tau + local_evap * pheromone_init
                        self.pheromone[path[-2], current] = updated
                        self.pheromone[current, path[-2]] = updated

                if path[-1] != goal_idx:
                    continue
                length = sum(self._dist(path[i], path[i + 1]) for i in range(len(path) - 1))
                if length < best_len:
                    best_len = length
                    best_path = path
                    best_backtracks = backtracks

            # pheromone evaporation + best update
            if self.pheromone is None:
                self.pheromone = np.ones((n, n), dtype=np.float64) * pheromone_init
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

        if self.last_target_idx is None:
            self.last_target_idx = target_idx
        elif target_idx != self.last_target_idx:
            # Invalidate cached path/state when the active target changes.
            self.current_path = []
            self.last_plan_slot = -1
            self.last_goal_point = None
            self.last_path_direct = False
            self.last_target_idx = target_idx

        if not self.current_path:
            replan_interval = int(self._param("replan_interval_steps", 1))
            slot_idx = int(state.get("slot_idx", -1))
            if (
                self.last_path_direct
                and self.last_target_idx == target_idx
                and self.last_goal_point is not None
                and replan_interval > 1
                and slot_idx >= 0
                and (slot_idx - self.last_plan_slot) < replan_interval
            ):
                return Action(target=self.last_goal_point, info={"status": "reuse", "target_idx": target_idx})
            start = self._nearest_reachable_node(pos)
            goal = self._nearest_node(target)
            node_path, backtracks = self._build_path(start, goal)
            self.current_path = [self.graph.nodes[i] for i in node_path]
            if not self.current_path:
                self.current_path = [target]
            else:
                last = self.current_path[-1]
                if (last[0] - target[0]) ** 2 + (last[1] - target[1]) ** 2 + (last[2] - target[2]) ** 2 > 1e-6:
                    self.current_path.append(target)
            self.last_plan_slot = int(state.get("slot_idx", -1))
            self.last_target_idx = target_idx
            self.last_goal_point = target
            self.last_path_direct = len(self.current_path) <= 1
            info = {"status": "ok", "target_idx": target_idx, "backtracks": backtracks}
        else:
            info = {"status": "ok", "target_idx": target_idx}

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
        return Action(target=next_wp, info=info)

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        target_idx = state.get("target_idx", 0)
        if target_idx >= len(targets):
            return {"status": "done", "path": []}
        target = targets[target_idx]
        start = self._nearest_reachable_node(pos)
        goal = self._nearest_node(target)
        node_path, backtracks = self._build_path(start, goal)
        path = [self.graph.nodes[i] for i in node_path]
        if not path:
            path = [target]
        else:
            last = path[-1]
            if (last[0] - target[0]) ** 2 + (last[1] - target[1]) ** 2 + (last[2] - target[2]) ** 2 > 1e-6:
                path.append(target)
        return {"status": "ok", "path": path, "target_idx": target_idx, "backtracks": backtracks}
