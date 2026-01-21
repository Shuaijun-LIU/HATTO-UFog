"""Simulation runner for baseline evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import math

from ufog_network.baselines.base import Action
from ufog_network.alloc import make_resource_allocator
from ufog_network.baselines import make_baseline
from ufog_network.config import Config
from ufog_network.env.generators import generate_world
from ufog_network.env.wind import WindField
from ufog_network.env.metrics import compute_metrics
from ufog_network.env.tasks import generate_md_positions, sample_tasks
from ufog_network.env.queue import QueueState
from ufog_network.io import export_world
from ufog_network.logging.parquet_logger import ParquetLogger
from ufog_network.logging.run_writer import make_run_id, prepare_run_dir, write_config, write_summary
from ufog_network.seeding import make_rng
from ufog_network.sim.dynamics import KinematicState, move_towards, movement_energy
from ufog_network.sim.energy import movement_energy_from_omegas
from ufog_network.sim.rigid_body import RigidBodyState, RigidBodyModel, euler_from_quat
from ufog_network.sim.control_loop import LowLevelController
from ufog_network.utils import shortest_path


@dataclass
class SimulationOutput:
    run_id: str
    summary: Dict[str, Any]
    output_dir: str


class Simulator:
    def __init__(self, cfg: Config, world=None) -> None:
        self.cfg = cfg
        self.world = world or generate_world(cfg.world)
        self.rng = make_rng(cfg.sim.seed)
        self.md_positions, self.md_service_positions = generate_md_positions(self.world, cfg.tasks, cfg.sim.seed)
        # Align md_count with sampled positions (PPP mode may vary)
        self.cfg.tasks.md_count = len(self.md_positions)
        self.queue_state = QueueState.init(len(self.md_positions)) if self.cfg.delay.queue_model == "backlog" else None
        self.waypoint_set = None
        if self.world.waypoints and self.world.waypoints.nodes:
            self.waypoint_set = {tuple(n) for n in self.world.waypoints.nodes}
        self.baseline = make_baseline(cfg.baseline)
        self.baseline.reset(self.world, cfg.sim.seed)
        self.resource_allocator = make_resource_allocator(cfg.resource)
        self.state = None
        self.rb_state = None
        self.rb_model = None
        self.low_level = None
        self.wind = WindField(cfg.wind) if cfg.wind.enabled else None
        self.current_wind = np.zeros(3, dtype=float)
        self.target_order = self._init_target_order()
        self.target_pos = 0
        self.target_idx = None
        self.target_steps = 0
        self.target_skip: Dict[int, int] = {}
        self.slot_idx = 0
        self.time_s = 0.0
        self.visited_md = set()
        self.all_md_visited = False
        self._init_uav_state()
        self._select_target_idx(initial=True)
        self.last_control = None
        if self.cfg.dynamics.mode == "rigid_body":
            self.rb_state = RigidBodyState(
                pos=self.state.pos.copy(),
                vel=self.state.vel.copy(),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                omega=np.zeros(3, dtype=float),
            )
            self.rb_model = RigidBodyModel(self.cfg.dynamics)
            self.low_level = LowLevelController(self.cfg.control, self.cfg.dynamics, self.cfg.sim)

    def _min_safe_altitude(self, x: float, y: float, margin: float) -> float:
        min_z = self.world.terrain.height(x, y) + self.world.terrain.clearance_m + margin
        for b in self.world.buildings:
            if abs(x - b.x) <= b.width / 2 and abs(y - b.y) <= b.depth / 2:
                min_z = max(min_z, b.base_z + b.height + margin)
        for o in self.world.obstacles:
            dx = x - o.x
            dy = y - o.y
            d_xy = math.sqrt(dx * dx + dy * dy)
            if d_xy < o.radius:
                z_clear = o.z + math.sqrt(max(0.0, o.radius * o.radius - d_xy * d_xy))
                min_z = max(min_z, z_clear + margin)
        return min_z

    def _hits_obstacle(self, point: Tuple[float, float, float]) -> bool:
        for b in self.world.buildings:
            if b.contains(point):
                return True
        for o in self.world.obstacles:
            if o.contains(point):
                return True
        return False

    def _init_target_order(self) -> List[int]:
        order = list(range(len(self.md_positions)))
        self.rng.shuffle(order)
        return order

    def _init_uav_state(self) -> None:
        z0 = self.world.terrain.height(0.0, 0.0) + self.world.terrain.clearance_m + self.cfg.sim.initial_altitude_margin_m
        pos = np.array([0.0, 0.0, z0], dtype=float)
        # Ensure free; fallback to random sampling
        if not self.world.is_free(tuple(pos)):
            for _ in range(self.cfg.sim.init_pos_attempts):
                x = (self.rng.random() - 0.5) * self.world.cfg.size_m
                y = (self.rng.random() - 0.5) * self.world.cfg.size_m
                z = self.world.terrain.height(x, y) + self.world.terrain.clearance_m + self.cfg.sim.initial_altitude_margin_m
                if self.world.is_free((x, y, z)):
                    pos = np.array([x, y, z], dtype=float)
                    break
        self.state = KinematicState(pos=pos, vel=np.zeros(3, dtype=float))
        self.battery_wh = self.cfg.energy.battery_wh

    def _targets(self) -> List[Tuple[float, float, float]]:
        return self.md_service_positions

    def _tick_target_skip(self) -> None:
        if not self.target_skip:
            return
        for key in list(self.target_skip.keys()):
            remaining = self.target_skip[key] - 1
            if remaining <= 0:
                del self.target_skip[key]
            else:
                self.target_skip[key] = remaining

    def _select_target_idx(self, initial: bool = False) -> Optional[int]:
        if len(self.md_positions) == 0:
            self.target_idx = None
            return None
        if len(self.visited_md) >= len(self.md_positions):
            self.all_md_visited = True
            self.target_idx = None
            return None
        policy = self.cfg.sim.target_policy
        if policy == "nearest_unserved":
            candidates = [
                i
                for i in range(len(self.md_positions))
                if i not in self.visited_md and self.target_skip.get(i, 0) <= 0
            ]
            if not candidates:
                candidates = [i for i in range(len(self.md_positions)) if i not in self.visited_md]
            if not candidates:
                self.all_md_visited = True
                self.target_idx = None
                return None
            px, py = float(self.state.pos[0]), float(self.state.pos[1])
            best = None
            best_d2 = 1e18
            for i in candidates:
                sx, sy, _ = self.md_service_positions[i]
                dx = sx - px
                dy = sy - py
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = i
            self.target_idx = best
            self.target_steps = 0
            return self.target_idx
        if policy != "shuffle":
            raise ValueError(f"Unknown target_policy: {policy}")
        if initial and not self.target_order:
            self.target_order = self._init_target_order()
        while True:
            while self.target_pos < len(self.target_order) and self.target_order[self.target_pos] in self.visited_md:
                self.target_pos += 1
            if self.target_pos < len(self.target_order):
                self.target_idx = self.target_order[self.target_pos]
                self.target_steps = 0
                return self.target_idx
            if len(self.visited_md) >= len(self.md_positions):
                self.all_md_visited = True
                self.target_idx = None
                return None
            self.target_order = self._init_target_order()
            self.target_pos = 0

    def _is_service_reached(self, md_idx: int) -> bool:
        if md_idx < 0 or md_idx >= len(self.md_service_positions):
            return False
        sx, sy, sz = self.md_service_positions[md_idx]
        if self.cfg.sim.service_radius_m > 0.0:
            dx = float(self.state.pos[0] - sx)
            dy = float(self.state.pos[1] - sy)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.cfg.sim.service_radius_m:
                return False
        else:
            dist = float(np.linalg.norm(self.state.pos - np.array((sx, sy, sz))))
            if dist > self.cfg.sim.reach_threshold_m:
                return False
        if self.cfg.sim.service_require_los:
            if not self.world.segment_is_free(tuple(self.state.pos), (sx, sy, sz), step=self.world.cfg.connect_step_m):
                return False
        return True

    def _advance_target_if_reached(self) -> None:
        if self.target_idx is None:
            return
        if self._is_service_reached(self.target_idx):
            self.visited_md.add(self.target_idx)
            if self.cfg.sim.target_policy == "shuffle":
                self.target_pos += 1
            self.target_idx = None
            self.target_steps = 0

    def _update_service_visits(self) -> None:
        for idx in range(len(self.md_positions)):
            if idx in self.visited_md:
                continue
            if self._is_service_reached(idx):
                self.visited_md.add(idx)
        if len(self.visited_md) >= len(self.md_positions):
            self.all_md_visited = True

    def _safe_target(self, target: Tuple[float, float, float], allow_graph_reroute: bool = True) -> Tuple[Tuple[float, float, float], bool]:
        cur = tuple(self.state.pos)
        x, y, z = target
        min_z = self._min_safe_altitude(x, y, 0.0)
        if min_z >= self.world.cfg.height_m:
            viol_alt = 1
            min_z = max(0.0, self.world.cfg.height_m - 1e-3)
        z = min(max(z, min_z), self.world.cfg.height_m)
        target = (x, y, z)
        max_step = max(0.5, self.cfg.sim.max_speed_m_s * self.cfg.sim.decision_dt_s * 0.5)
        if allow_graph_reroute:
            check_step = min(self.world.cfg.connect_step_m, max_step)
        else:
            # Looser collision check for waypoint-following to reduce false reroutes.
            check_step = self.world.cfg.connect_step_m
        if self.world.segment_is_free(cur, target, step=check_step):
            return target, False
        # Try to approach above the target (vertical descent), if feasible.
        x, y, z = target
        approach_z = self._min_safe_altitude(x, y, self.cfg.sim.reroute_altitude_margin_m)
        flyover_z = self.world.cfg.height_m - self.cfg.sim.flyover_altitude_margin_m
        approach_z = min(self.world.cfg.height_m, max(z, approach_z, flyover_z))
        approach = (x, y, approach_z)
        if self.world.segment_is_free(cur, approach, step=check_step):
            return approach, True
        # Try climbing in place to a flyover altitude to clear obstacles.
        if flyover_z > cur[2] + 1e-3:
            climb = (cur[0], cur[1], flyover_z)
            if self.world.segment_is_free(cur, climb, step=check_step):
                return climb, True
        route_target = approach if approach_z > z else target
        # Try waypoint graph routing if available
        if allow_graph_reroute and self.world.waypoints and self.world.waypoints.nodes:
            graph = self.world.waypoints
            nodes = graph.nodes
            # Pick a reachable start node from current position
            candidates = sorted(
                range(len(nodes)),
                key=lambda i: (nodes[i][0] - cur[0]) ** 2 + (nodes[i][1] - cur[1]) ** 2 + (nodes[i][2] - cur[2]) ** 2,
            )
            start = None
            for idx in candidates[: min(60, len(candidates))]:
                if self.world.segment_is_free(cur, nodes[idx], step=check_step):
                    start = idx
                    break
            if start is None:
                # No reachable node from current position: climb in place to clear obstacles.
                safe_z = self._min_safe_altitude(cur[0], cur[1], self.cfg.sim.reroute_altitude_margin_m)
                return (cur[0], cur[1], max(cur[2], safe_z)), True
            goal = self._nearest_node(route_target)
            path = shortest_path(nodes, graph.edges, start, goal)
            if path and len(path) > 1:
                next_node = nodes[path[1]]
                if self.world.segment_is_free(cur, next_node, step=check_step):
                    return next_node, True
                return nodes[start], True
            return nodes[start], True
        # Fallback: lift altitude above terrain
        x, y, z = target
        safe_z = self._min_safe_altitude(x, y, self.cfg.sim.reroute_altitude_margin_m)
        return (x, y, max(z, safe_z)), True

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

    def _wind_vector(self) -> np.ndarray | None:
        if self.wind is None:
            return None
        apply_to = self.cfg.wind.apply_to
        if apply_to not in ("both", self.cfg.dynamics.mode):
            return None
        return self.wind.sample(tuple(self.state.pos), self.time_s)

    def _step_kinematic(
        self, target: Tuple[float, float, float], wind_vec: np.ndarray | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        new_state, dist, speed, speed_violation = move_towards(
            self.state,
            target,
            dt=self.cfg.sim.decision_dt_s,
            max_speed=self.cfg.sim.max_speed_m_s,
            max_accel=self.cfg.sim.max_accel_m_s2,
        )
        if wind_vec is not None:
            wind_accel = self.cfg.wind.accel_gain * (wind_vec - new_state.vel)
            new_vel = new_state.vel + wind_accel * self.cfg.sim.decision_dt_s
            new_pos = self.state.pos + new_vel * self.cfg.sim.decision_dt_s
            new_state = KinematicState(pos=new_pos, vel=new_vel)
            dist = float(np.linalg.norm(new_pos - self.state.pos))
            speed = float(np.linalg.norm(new_vel))
            speed_violation = speed > self.cfg.sim.max_speed_m_s + 1e-3
        pos_tuple = tuple(new_state.pos)
        viol_obstacle = 0
        viol_alt = 0
        x, y, z = pos_tuple
        min_z = self._min_safe_altitude(x, y, 0.0)
        if min_z >= self.world.cfg.height_m:
            viol_alt = 1
            min_z = max(0.0, self.world.cfg.height_m - 1e-3)
        if z > self.world.cfg.height_m:
            viol_alt = 1
            new_state.pos[2] = self.world.cfg.height_m
        if z <= min_z:
            viol_alt = 1
            new_state.pos[2] = min_z + 1e-3
        pos_tuple = tuple(new_state.pos)
        if self._hits_obstacle(pos_tuple):
            viol_obstacle = 1
            safe_z = self._min_safe_altitude(x, y, self.cfg.sim.recovery_altitude_margin_m)
            new_state.pos[2] = max(new_state.pos[2], safe_z)
            if self._hits_obstacle(tuple(new_state.pos)):
                new_state = self.state
        self.state = new_state
        self.last_control = None
        meta = {
            "speed": speed,
            "speed_violation": speed_violation,
            "viol_obstacle": viol_obstacle,
            "viol_alt": viol_alt,
            "E_mov": movement_energy(dist, speed, self.cfg.sim.decision_dt_s, self.cfg.sim),
        }
        return {"pos": tuple(new_state.pos), "vel": tuple(new_state.vel)}, meta

    def _step_rigidbody(
        self, target: Tuple[float, float, float], wind_vec: np.ndarray | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        steps = max(1, int(self.cfg.sim.decision_dt_s / max(1e-6, self.cfg.sim.control_dt_s)))
        dt = self.cfg.sim.decision_dt_s / steps
        speed_violation = False
        viol_obstacle = 0
        viol_alt = 0
        E_mov = 0.0
        last_ctrl = None
        for _ in range(steps):
            prev_state = RigidBodyState(
                pos=self.rb_state.pos.copy(),
                vel=self.rb_state.vel.copy(),
                quat=self.rb_state.quat.copy(),
                omega=self.rb_state.omega.copy(),
            )
            ctrl = self.low_level.step(self.rb_state, target, dt)
            prev_pos = self.rb_state.pos.copy()
            self.rb_state, _thrust, _tau = self.rb_model.step(
                self.rb_state,
                ctrl.omegas,
                dt,
                wind_vel=wind_vec,
                wind_accel_gain=self.cfg.wind.accel_gain if wind_vec is not None else 0.0,
            )
            step_dist = float(np.linalg.norm(self.rb_state.pos - prev_pos))
            speed = float(np.linalg.norm(self.rb_state.vel))
            if speed > self.cfg.sim.max_speed_m_s + 1e-3:
                speed_violation = True
            E_mov += movement_energy_from_omegas(ctrl.omegas, dt, self.cfg.energy, self.cfg.dynamics)
            pos_tuple = tuple(self.rb_state.pos)
            x, y, z = pos_tuple
            min_z = self._min_safe_altitude(x, y, 0.0)
            if z > self.world.cfg.height_m:
                viol_alt = 1
                self.rb_state.pos[2] = self.world.cfg.height_m
            if z <= min_z:
                viol_alt = 1
                self.rb_state.pos[2] = min_z + 1e-3
            pos_tuple = tuple(self.rb_state.pos)
            if self._hits_obstacle(pos_tuple):
                viol_obstacle = 1
                safe_z = self._min_safe_altitude(x, y, self.cfg.sim.recovery_altitude_margin_m)
                self.rb_state.pos[2] = max(self.rb_state.pos[2], safe_z)
                pos_tuple = tuple(self.rb_state.pos)
                if self._hits_obstacle(pos_tuple):
                    self.rb_state = prev_state
                    pos_tuple = tuple(self.rb_state.pos)
            last_ctrl = ctrl
        # Sync kinematic state view
        self.state = KinematicState(pos=self.rb_state.pos.copy(), vel=self.rb_state.vel.copy())
        self.last_control = last_ctrl
        meta = {
            "speed": float(np.linalg.norm(self.rb_state.vel)),
            "speed_violation": speed_violation,
            "viol_obstacle": viol_obstacle,
            "viol_alt": viol_alt,
            "E_mov": E_mov,
        }
        return {"pos": tuple(self.rb_state.pos), "vel": tuple(self.rb_state.vel)}, meta

    def step(self, external_action: Optional[Action] = None) -> Tuple[Dict[str, Any], bool]:
        if self.slot_idx >= self.cfg.sim.steps:
            return {}, True

        self._tick_target_skip()
        if self.target_idx is None:
            self._select_target_idx()
        tasks = sample_tasks(self.cfg.tasks, len(self.md_positions), self.cfg.sim.seed + self.slot_idx)
        arrival_rate = self.cfg.tasks.arrival_rate
        if self.cfg.tasks.arrival_rate_mode == "sampled":
            per_md = {j: 0 for j in range(len(self.md_positions))}
            for t in tasks:
                per_md[t.md_id] = per_md.get(t.md_id, 0) + 1
            L = self.cfg.sim.decision_dt_s
            arrival_rate = {j: per_md[j] / max(1e-6, L) for j in per_md}
        targets = self._targets()
        state = {
            "uav_pos": tuple(self.state.pos),
            "uav_vel": tuple(self.state.vel),
            "targets": targets,
            "target_idx": self.target_idx if self.target_idx is not None else len(targets),
            "md_positions": self.md_positions,
            "md_service_positions": self.md_service_positions,
            "tasks": tasks,
            "battery_wh": self.battery_wh,
            "battery_wh_max": self.cfg.energy.battery_wh,
            "slot_idx": self.slot_idx,
            "time_s": self.time_s,
        }

        action = external_action or self.baseline.act(state)
        target_idx_used = self.target_idx if self.target_idx is not None else None
        try:
            action_target_idx = action.info.get("target_idx") if isinstance(action.info, dict) else None
        except Exception:
            action_target_idx = None
        if action_target_idx is not None:
            try:
                action_target_idx_int = int(action_target_idx)
            except Exception:
                action_target_idx_int = None
            if action_target_idx_int is not None and 0 <= action_target_idx_int < len(self.md_positions):
                target_idx_used = action_target_idx_int

        md_target_ground = None
        md_target_service = None
        if target_idx_used is not None and 0 <= target_idx_used < len(self.md_positions):
            md_target_ground = self.md_positions[target_idx_used]
            md_target_service = self.md_service_positions[target_idx_used]

        target = action.target
        graph_baselines = {"acs", "acs_ds", "cps_aco", "ga_sca"}
        allow_graph_reroute = self.cfg.baseline.name not in graph_baselines
        target, reroute = self._safe_target(target, allow_graph_reroute=allow_graph_reroute)
        if reroute and hasattr(self.baseline, "current_path"):
            try:
                self.baseline.current_path = []
            except Exception:
                pass

        wind_vec = self._wind_vector()
        if wind_vec is None:
            self.current_wind = np.zeros(3, dtype=float)
        else:
            self.current_wind = wind_vec
        if self.cfg.dynamics.mode == "rigid_body":
            _, meta = self._step_rigidbody(target, wind_vec)
        else:
            _, meta = self._step_kinematic(target, wind_vec)

        speed_violation = meta["speed_violation"]
        viol_obstacle = meta["viol_obstacle"]
        viol_alt = meta["viol_alt"]
        E_mov = meta["E_mov"]
        alloc = None
        if self.resource_allocator is not None:
            alloc = self.resource_allocator.allocate(
                tasks=tasks,
                md_positions=self.md_positions,
                uav_pos=tuple(self.state.pos),
                cfg=self.cfg,
                world=self.world,
                E_mov=E_mov,
            )
        metrics = compute_metrics(
            tasks=tasks,
            md_positions=self.md_positions,
            uav_pos=tuple(self.state.pos),
            energy_cfg=self.cfg.energy,
            comm_cfg=self.cfg.comm,
            delay_cfg=self.cfg.delay,
            sim_cfg=self.cfg.sim,
            E_mov=E_mov,
            world=self.world,
            offload_cfg=self.cfg.offload,
            cloud_cfg=self.cfg.cloud,
            arrival_rate=arrival_rate,
            alloc=alloc,
            queue_state=self.queue_state,
            offload_granularity=self.cfg.resource.offload_granularity,
        )
        S = metrics.D_total + self.cfg.sim.epsilon * metrics.E_total

        # Update battery
        self.battery_wh -= metrics.E_total / 3600.0
        viol_battery = 1 if self.battery_wh <= 0.0 else 0

        roll = None
        pitch = None
        yaw = None
        omega_x = None
        omega_y = None
        omega_z = None
        rotor_1 = None
        rotor_2 = None
        rotor_3 = None
        rotor_4 = None
        thrust = None
        tau_x = None
        tau_y = None
        tau_z = None
        viol_att = 0
        viol_angrate = 0
        viol_rotor = 0
        if self.cfg.dynamics.mode == "rigid_body" and self.rb_state is not None:
            roll, pitch, yaw = euler_from_quat(self.rb_state.quat)
            omega_x, omega_y, omega_z = (float(self.rb_state.omega[0]), float(self.rb_state.omega[1]), float(self.rb_state.omega[2]))
            max_tilt = math.radians(self.cfg.control.max_tilt_deg)
            if abs(roll) > max_tilt or abs(pitch) > max_tilt:
                viol_att = 1
            if max(abs(self.rb_state.omega[0]), abs(self.rb_state.omega[1]), abs(self.rb_state.omega[2])) > self.cfg.dynamics.max_omega_rad_s:
                viol_angrate = 1
            if self.last_control is not None:
                rotor_1 = float(self.last_control.omegas[0])
                rotor_2 = float(self.last_control.omegas[1])
                rotor_3 = float(self.last_control.omegas[2])
                rotor_4 = float(self.last_control.omegas[3])
                thrust = float(self.last_control.thrust)
                tau_x = float(self.last_control.tau[0])
                tau_y = float(self.last_control.tau[1])
                tau_z = float(self.last_control.tau[2])
                viol_rotor = 1 if self.last_control.saturated else 0

        # Update state and counters
        # self.state updated in kinematic/rigid-body steps
        self._advance_target_if_reached()
        prev_target = self.target_idx
        self._update_service_visits()
        if prev_target is not None and prev_target in self.visited_md:
            if self.cfg.sim.target_policy == "shuffle":
                self.target_pos += 1
            self.target_idx = None
            self.target_steps = 0
        if self.target_idx is None:
            self._select_target_idx()
        if self.target_idx is not None:
            self.target_steps += 1
            if self.cfg.sim.target_max_steps > 0 and self.target_steps >= self.cfg.sim.target_max_steps:
                if self.cfg.sim.target_skip_steps > 0:
                    self.target_skip[self.target_idx] = self.cfg.sim.target_skip_steps
                if self.cfg.sim.target_policy == "shuffle":
                    self.target_pos += 1
                self.target_idx = None
                self.target_steps = 0
                self._select_target_idx()
        self.slot_idx += 1
        self.time_s += self.cfg.sim.decision_dt_s
        pos_error = float(np.linalg.norm(self.state.pos - np.array(target)))
        pos_error_md = (
            float(np.linalg.norm(self.state.pos - np.array(md_target_ground))) if md_target_ground is not None else None
        )
        if md_target_service is None:
            pos_error_md_service = None
        else:
            dx = float(self.state.pos[0] - md_target_service[0])
            dy = float(self.state.pos[1] - md_target_service[1])
            if self.cfg.sim.service_radius_m > 0.0:
                # Match the reach criterion (horizontal service radius).
                pos_error_md_service = math.sqrt(dx * dx + dy * dy)
            else:
                pos_error_md_service = float(np.linalg.norm(self.state.pos - np.array(md_target_service)))
        wind_speed = float(np.linalg.norm(self.current_wind))

        row = {
            "run_id": None,
            "seed": self.cfg.sim.seed,
            "slot_idx": self.slot_idx - 1,
            "time_s": self.time_s,
            "x": float(self.state.pos[0]),
            "y": float(self.state.pos[1]),
            "z": float(self.state.pos[2]),
            "vx": float(self.state.vel[0]),
            "vy": float(self.state.vel[1]),
            "vz": float(self.state.vel[2]),
            "battery_wh": self.battery_wh,
            "target_idx": int(target_idx_used) if target_idx_used is not None else -1,
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "md_target_x": md_target_ground[0] if md_target_ground is not None else None,
            "md_target_y": md_target_ground[1] if md_target_ground is not None else None,
            "md_target_z": md_target_ground[2] if md_target_ground is not None else None,
            "md_service_x": md_target_service[0] if md_target_service is not None else None,
            "md_service_y": md_target_service[1] if md_target_service is not None else None,
            "md_service_z": md_target_service[2] if md_target_service is not None else None,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "omega_x": omega_x,
            "omega_y": omega_y,
            "omega_z": omega_z,
            "dyn_mode": self.cfg.dynamics.mode,
            "rotor_omega_1": rotor_1,
            "rotor_omega_2": rotor_2,
            "rotor_omega_3": rotor_3,
            "rotor_omega_4": rotor_4,
            "thrust": thrust,
            "tau_x": tau_x,
            "tau_y": tau_y,
            "tau_z": tau_z,
            "wind_x": float(self.current_wind[0]),
            "wind_y": float(self.current_wind[1]),
            "wind_z": float(self.current_wind[2]),
            "wind_speed": wind_speed,
            "pos_error": pos_error,
            "pos_error_md": pos_error_md,
            "pos_error_md_service": pos_error_md_service,
            "E_total": metrics.E_total,
            "E_mov": metrics.E_mov,
            "E_tr": metrics.E_tr,
            "E_comp": metrics.E_comp,
            "D_total": metrics.D_total,
            "D_tr": metrics.D_tr,
            "D_comp": metrics.D_comp,
            "D_q": metrics.D_q,
            "D_uavq": metrics.D_uavq,
            "S": S,
            "viol_speed": int(speed_violation),
            "viol_alt": int(viol_alt),
            "viol_battery": int(viol_battery),
            "viol_obstacle": int(viol_obstacle),
            "viol_att": int(viol_att),
            "viol_angrate": int(viol_angrate),
            "viol_rotor": int(viol_rotor),
            "reroute": int(reroute),
        }
        done = self.slot_idx >= self.cfg.sim.steps or viol_battery == 1
        if self.cfg.sim.stop_when_all_md_visited and self.all_md_visited:
            done = True
        return row, done

    def run(self, output_root: str, export_world_json: bool = True) -> SimulationOutput:
        run_id = make_run_id(self.cfg.baseline.name)
        run_dir = prepare_run_dir(output_root, run_id)
        write_config(run_dir, self.cfg)
        if export_world_json:
            export_world(self.world, str(run_dir / "world.json"))

        logger = ParquetLogger(str(run_dir / "timeseries.parquet"))
        totals = {
            "E_total": 0.0,
            "E_mov": 0.0,
            "E_tr": 0.0,
            "E_comp": 0.0,
            "D_total": 0.0,
            "D_tr": 0.0,
            "D_comp": 0.0,
            "D_q": 0.0,
            "D_uavq": 0.0,
            "S": 0.0,
            "viol_speed": 0,
            "viol_alt": 0,
            "viol_battery": 0,
            "viol_obstacle": 0,
            "viol_att": 0,
            "viol_angrate": 0,
            "viol_rotor": 0,
        }
        while True:
            row, done = self.step()
            if not row:
                break
            row["run_id"] = run_id
            logger.append(row)
            totals["E_total"] += row["E_total"]
            totals["E_mov"] += row["E_mov"]
            totals["E_tr"] += row["E_tr"]
            totals["E_comp"] += row["E_comp"]
            totals["D_total"] += row["D_total"]
            totals["D_tr"] += row["D_tr"]
            totals["D_comp"] += row["D_comp"]
            totals["D_q"] += row["D_q"]
            totals["D_uavq"] += row["D_uavq"]
            totals["S"] += row["S"]
            totals["viol_speed"] += row["viol_speed"]
            totals["viol_alt"] += row["viol_alt"]
            totals["viol_battery"] += row["viol_battery"]
            totals["viol_obstacle"] += row["viol_obstacle"]
            totals["viol_att"] += row["viol_att"]
            totals["viol_angrate"] += row["viol_angrate"]
            totals["viol_rotor"] += row["viol_rotor"]
            if done:
                break

        logger.flush()
        summary = {
            "run_id": run_id,
            "steps": self.slot_idx,
            "battery_wh_end": self.battery_wh,
            "baseline": self.cfg.baseline.name,
            "totals": totals,
            "all_md_visited": bool(self.all_md_visited),
            "unique_md_visited": int(len(self.visited_md)),
        }
        write_summary(run_dir, summary)
        return SimulationOutput(run_id=run_id, summary=summary, output_dir=str(run_dir))
