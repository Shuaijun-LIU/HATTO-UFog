"""GA-SCA baseline: GA-based target ordering with local refinement."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from ufog_network.baselines.base import Action, Baseline
from ufog_network.seeding import make_rng
from ufog_network.utils import shortest_path


class GASCABaseline(Baseline):
    name = "ga_sca"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def reset(self, world, seed: int) -> None:
        super().reset(world, seed)
        self.rng = make_rng(seed)
        self.sequence: List[int] = []
        self.sequence_idx = 0
        self.current_path: List[Tuple[float, float, float]] = []
        self._targets_cache: List[Tuple[float, float, float]] = []

    def _segment_cost(self, p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> float:
        dist = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
        if not self.world.segment_is_free(p0, p1, step=self.world.cfg.connect_step_m):
            return dist + 1e6
        threat = 0.0
        for o in self.world.obstacles:
            # approximate threat by distance to obstacle center
            dx = p1[0] - o.x
            dy = p1[1] - o.y
            dz = p1[2] - o.z
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            threat += 1.0 / (d + 1e-6)
        w_threat = float(self.params.get("threat_weight", self.params.get("w_threat", 20.0)))
        return dist + w_threat * threat

    def _route_cost(self, order: List[int], start: Tuple[float, float, float], targets: List[Tuple[float, float, float]]) -> float:
        cost = 0.0
        cur = start
        for idx in order:
            tgt = targets[idx]
            cost += self._segment_cost(cur, tgt)
            cur = tgt
        return cost

    def _init_population(self, n: int) -> List[List[int]]:
        pop = []
        base = list(range(n))
        for _ in range(self.params.get("population", 40)):
            perm = base[:]
            self.rng.shuffle(perm)
            pop.append(perm)
        return pop

    def _tournament(self, pop: List[List[int]], scores: List[float]) -> List[int]:
        k = int(self.params.get("tournament", 3))
        best = None
        best_score = 1e18
        for _ in range(k):
            idx = self.rng.integers(0, len(pop))
            if scores[idx] < best_score:
                best_score = scores[idx]
                best = pop[idx]
        return best[:]

    def _crossover(self, a: List[int], b: List[int]) -> List[int]:
        n = len(a)
        if n <= 2:
            return a[:]
        i, j = sorted(self.rng.choice(n, size=2, replace=False))
        child = [-1] * n
        child[i:j] = a[i:j]
        fill = [x for x in b if x not in child]
        ptr = 0
        for k in range(n):
            if child[k] == -1:
                child[k] = fill[ptr]
                ptr += 1
        return child

    def _mutate(self, seq: List[int]) -> None:
        if self.rng.random() < float(self.params.get("mutation", 0.2)):
            i, j = self.rng.choice(len(seq), size=2, replace=False)
            seq[i], seq[j] = seq[j], seq[i]

    def _sca_refine(self, order: List[int], start: Tuple[float, float, float], targets: List[Tuple[float, float, float]]) -> List[int]:
        improved = True
        best_order = order[:]
        best_cost = self._route_cost(order, start, targets)
        max_iters = int(self.params.get("sca_iters", 80))
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(1, len(best_order) - 1):
                for j in range(i + 1, len(best_order)):
                    candidate = best_order[:]
                    candidate[i:j] = reversed(candidate[i:j])
                    cost = self._route_cost(candidate, start, targets)
                    if cost < best_cost:
                        best_cost = cost
                        best_order = candidate
                        improved = True
        return best_order

    def _optimize_sequence(self, start: Tuple[float, float, float], targets: List[Tuple[float, float, float]]) -> List[int]:
        n = len(targets)
        if n <= 1:
            return list(range(n))
        pop = self._init_population(n)
        generations = int(self.params.get("generations", 50))
        elite = int(self.params.get("elite", 2))
        for _ in range(generations):
            scores = [self._route_cost(ind, start, targets) for ind in pop]
            order = np.argsort(scores)
            new_pop = [pop[i] for i in order[:elite]]
            while len(new_pop) < len(pop):
                parent_a = self._tournament(pop, scores)
                parent_b = self._tournament(pop, scores)
                child = self._crossover(parent_a, parent_b)
                self._mutate(child)
                new_pop.append(child)
            pop = new_pop
        scores = [self._route_cost(ind, start, targets) for ind in pop]
        best_idx = int(np.argmin(scores))
        best = pop[best_idx]
        return self._sca_refine(best, start, targets)

    def _build_path(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        if not self.world.waypoints or not self.world.waypoints.nodes:
            return [goal]
        graph = self.world.waypoints
        start_idx = None
        candidates = sorted(
            range(len(graph.nodes)),
            key=lambda i: (graph.nodes[i][0] - start[0]) ** 2 + (graph.nodes[i][1] - start[1]) ** 2 + (graph.nodes[i][2] - start[2]) ** 2,
        )
        for idx in candidates[: min(60, len(candidates))]:
            if self.world.segment_is_free(start, graph.nodes[idx], step=self.world.cfg.connect_step_m):
                start_idx = idx
                break
        if start_idx is None:
            return [goal]
        goal_idx = self._nearest_node(goal)
        path = shortest_path(graph.nodes, graph.edges, start_idx, goal_idx)
        if not path:
            return [goal]
        coords = [graph.nodes[i] for i in path[1:]]
        if not coords:
            return [goal]
        last = coords[-1]
        if (last[0] - goal[0]) ** 2 + (last[1] - goal[1]) ** 2 + (last[2] - goal[2]) ** 2 > 1e-6:
            coords.append(goal)
        return coords

    def _nearest_node(self, point: Tuple[float, float, float]) -> int:
        nodes = self.world.waypoints.nodes
        best_idx = 0
        best_d = 1e18
        for i, n in enumerate(nodes):
            d = (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2 + (n[2] - point[2]) ** 2
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx

    def act(self, state: Dict[str, Any]) -> Action:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        if not targets:
            return Action(target=pos, info={"status": "done"})

        if not self.sequence or targets != self._targets_cache:
            # Optionally limit target count using task load
            max_targets = int(self.params.get("max_targets", len(targets)))
            if "tasks" in state and max_targets < len(targets):
                task_load = {i: 0.0 for i in range(len(targets))}
                for t in state["tasks"]:
                    task_load[t.md_id] = task_load.get(t.md_id, 0.0) + t.size_bits
                ranked = sorted(task_load.items(), key=lambda x: x[1], reverse=True)
                selected = [idx for idx, _ in ranked[:max_targets]]
            else:
                selected = list(range(len(targets)))
            sub_targets = [targets[i] for i in selected]
            order = self._optimize_sequence(pos, sub_targets)
            self.sequence = [selected[i] for i in order]
            self.sequence_idx = 0
            self._targets_cache = targets[:]

        if self.sequence_idx >= len(self.sequence):
            return Action(target=pos, info={"status": "done"})

        target_idx = self.sequence[self.sequence_idx]
        goal = targets[target_idx]

        if not self.current_path:
            self.current_path = self._build_path(pos, goal)
            if not self.current_path:
                self.current_path = [goal]

        reach = float(self.params.get("goal_reach_m", 8.0))
        while self.current_path:
            next_wp = self.current_path[0]
            if math.sqrt((pos[0] - next_wp[0]) ** 2 + (pos[1] - next_wp[1]) ** 2 + (pos[2] - next_wp[2]) ** 2) <= reach:
                self.current_path.pop(0)
                continue
            break
        if not self.current_path:
            next_wp = goal
        else:
            next_wp = self.current_path[0]
        # Advance sequence if close to goal
        if math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2 + (pos[2] - goal[2]) ** 2) < reach:
            self.sequence_idx += 1
            self.current_path = []

        return Action(target=next_wp, info={"status": "ok", "target_idx": target_idx})

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pos = state["uav_pos"]
        targets: List[Tuple[float, float, float]] = state["targets"]
        if not targets:
            return {"status": "done", "sequence": [], "path": []}
        if not self.sequence or targets != self._targets_cache:
            max_targets = int(self.params.get("max_targets", len(targets)))
            if "tasks" in state and max_targets < len(targets):
                task_load = {i: 0.0 for i in range(len(targets))}
                for t in state["tasks"]:
                    task_load[t.md_id] = task_load.get(t.md_id, 0.0) + t.size_bits
                ranked = sorted(task_load.items(), key=lambda x: x[1], reverse=True)
                selected = [idx for idx, _ in ranked[:max_targets]]
            else:
                selected = list(range(len(targets)))
            sub_targets = [targets[i] for i in selected]
            order = self._optimize_sequence(pos, sub_targets)
            self.sequence = [selected[i] for i in order]
            self.sequence_idx = 0
            self._targets_cache = targets[:]
        target_idx = self.sequence[self.sequence_idx]
        goal = targets[target_idx]
        path = self._build_path(pos, goal)
        return {"status": "ok", "sequence": self.sequence[:], "path": path, "target_idx": target_idx}
