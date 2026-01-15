"""PSO-based resource allocation."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math
import numpy as np

from ufog_network.alloc.base import AllocationDecision, ResourceAllocator
from ufog_network.config import ResourceConfig
from ufog_network.env.metrics import compute_metrics, gamma_channel_allocation, uniform_channel_allocation, round_channels
from ufog_network.seeding import make_rng


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _offload_fractions(offload_cfg) -> Tuple[float, float, float]:
    if offload_cfg.mode == "md":
        return 0.0, 1.0, 0.0
    if offload_cfg.mode == "dc":
        return 0.0, 0.0, 1.0
    if offload_cfg.mode == "mixed":
        total = max(1e-9, offload_cfg.mixed_ratio_uav + offload_cfg.mixed_ratio_md + offload_cfg.mixed_ratio_dc)
        return offload_cfg.mixed_ratio_uav / total, offload_cfg.mixed_ratio_md / total, offload_cfg.mixed_ratio_dc / total
    return 1.0, 0.0, 0.0


def _offload_heuristic(total_size: float, distance: float, blocked: bool, offload_cfg, cloud_cfg) -> Tuple[float, float, float]:
    if blocked and offload_cfg.heuristic_blocked_to_md:
        return 0.0, 1.0, 0.0
    if cloud_cfg is not None and cloud_cfg.enabled and total_size >= offload_cfg.heuristic_dc_size_bits:
        return 0.0, 0.0, 1.0
    if distance >= offload_cfg.heuristic_distance_m or total_size >= offload_cfg.heuristic_size_bits:
        return 0.0, 1.0, 0.0
    return 1.0, 0.0, 0.0


def _power_default(comm_cfg) -> float:
    if comm_cfg.power_mode == "fixed":
        return comm_cfg.p_fixed_w
    return comm_cfg.p_max_w


class PSOAllocator(ResourceAllocator):
    def __init__(self, cfg: ResourceConfig) -> None:
        self.cfg = cfg
        self.rng = make_rng(cfg.pso.seed)

    def allocate(
        self,
        tasks: List[Any],
        md_positions: List[Tuple[float, float, float]],
        uav_pos: Tuple[float, float, float],
        cfg: Any,
        world: Any | None = None,
        E_mov: float = 0.0,
    ) -> AllocationDecision | None:
        if not tasks or not md_positions:
            return AllocationDecision(meta={"status": "no_tasks"})

        K = len(md_positions)
        size_sum: Dict[int, float] = {}
        for t in tasks:
            size_sum[t.md_id] = size_sum.get(t.md_id, 0.0) + t.size_bits

        optimize_offload = bool(self.cfg.optimize_offload)
        optimize_power = bool(self.cfg.optimize_power)
        optimize_freq = bool(self.cfg.optimize_freq)
        optimize_channels = bool(self.cfg.optimize_channels)
        offload_strategy = str(self.cfg.offload_strategy)

        # Precompute channel allocation if PSO does not optimize it
        channels = {}
        if not optimize_channels:
            if cfg.comm.channel_mode == "uniform":
                channels_raw = uniform_channel_allocation(size_sum, cfg.comm)
            else:
                channels_raw = gamma_channel_allocation(size_sum, cfg.comm)
            channels = round_channels(channels_raw, cfg.comm)

        if optimize_offload and offload_strategy == "hard":
            off_dim = 3 * K
        else:
            off_dim = 2 * K if optimize_offload else 0
        pow_dim = K if optimize_power else 0
        freq_dim = (K + 1) if optimize_freq else 0  # uav + per-md
        chan_dim = K if optimize_channels else 0
        dim = off_dim + pow_dim + freq_dim + chan_dim

        # Fallback if nothing to optimize
        if dim == 0:
            frac_uav, frac_md, frac_dc = _offload_fractions(cfg.offload)
            return AllocationDecision(
                channel_alloc=channels,
                power_w={j: _power_default(cfg.comm) for j in range(K)},
                freq_md_hz={j: cfg.energy.md_cpu_hz for j in range(K)},
                freq_uav_hz=cfg.energy.uav_cpu_hz,
                offload_uav={j: frac_uav for j in range(K)},
                offload_md={j: frac_md for j in range(K)},
                offload_dc={j: frac_dc for j in range(K)},
                meta={"status": "fixed"},
            )

        particles = max(1, int(self.cfg.pso.particles))
        iters = max(1, int(self.cfg.pso.iterations))
        inertia = float(self.cfg.pso.inertia)
        c1 = float(self.cfg.pso.c1)
        c2 = float(self.cfg.pso.c2)
        vel_clip = float(self.cfg.pso.vel_clip)
        tol = float(self.cfg.pso.tol)
        no_improve_limit = int(self.cfg.pso.no_improve_iters)

        pos = self.rng.uniform(-1.0, 1.0, size=(particles, dim))
        vel = self.rng.uniform(-0.1, 0.1, size=(particles, dim))

        fallback_offload: Dict[int, Tuple[float, float, float]] = {}
        if not optimize_offload:
            if cfg.offload.mode == "heuristic":
                for j, total_size in size_sum.items():
                    md_x, md_y, md_z = md_positions[j]
                    dx = uav_pos[0] - md_x
                    dy = uav_pos[1] - md_y
                    dz = uav_pos[2] - md_z
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    blocked = False
                    if world is not None and cfg.comm.enable_los:
                        blocked = not world.segment_is_free(uav_pos, (md_x, md_y, md_z), step=world.cfg.connect_step_m)
                    fallback_offload[j] = _offload_heuristic(total_size, dist, blocked, cfg.offload, cfg.cloud)
            else:
                frac_uav, frac_md, frac_dc = _offload_fractions(cfg.offload)
                for j in range(K):
                    fallback_offload[j] = (frac_uav, frac_md, frac_dc)

        def decode(vec: np.ndarray) -> AllocationDecision:
            idx = 0
            decision = AllocationDecision()

            if optimize_channels:
                raw = vec[idx : idx + chan_dim]
                weights = _softmax(raw, axis=0)
                total_channels = float(cfg.comm.channel_count)
                min_ch = float(cfg.comm.channel_min)
                if min_ch > 0 and min_ch * K <= total_channels:
                    remaining = total_channels - min_ch * K
                    decision.channel_alloc = {j: float(min_ch + remaining * weights[j]) for j in range(K)}
                else:
                    decision.channel_alloc = {j: float(total_channels * weights[j]) for j in range(K)}
                idx += chan_dim
            else:
                decision.channel_alloc = channels

            if optimize_offload:
                if offload_strategy == "hard":
                    raw = vec[idx : idx + off_dim].reshape(K, 3)
                    weights = _softmax(raw, axis=1)
                    for j in range(K):
                        choice = int(np.argmax(weights[j]))
                        decision.offload_uav[j] = 1.0 if choice == 0 else 0.0
                        decision.offload_md[j] = 1.0 if choice == 1 else 0.0
                        decision.offload_dc[j] = 1.0 if choice == 2 else 0.0
                else:
                    raw = vec[idx : idx + off_dim].reshape(K, 2)
                    raw_sig = _sigmoid(raw)
                    for j in range(K):
                        u = float(raw_sig[j, 0])
                        v = float(raw_sig[j, 1])
                        frac_uav = u
                        frac_md = (1.0 - u) * v
                        frac_dc = max(0.0, 1.0 - frac_uav - frac_md)
                        decision.offload_uav[j] = frac_uav
                        decision.offload_md[j] = frac_md
                        decision.offload_dc[j] = frac_dc
                idx += off_dim
            else:
                for j in range(K):
                    frac_uav, frac_md, frac_dc = fallback_offload.get(j, _offload_fractions(cfg.offload))
                    decision.offload_uav[j] = frac_uav
                    decision.offload_md[j] = frac_md
                    decision.offload_dc[j] = frac_dc

            if optimize_power:
                raw = vec[idx : idx + pow_dim]
                sig = _sigmoid(raw)
                for j in range(K):
                    decision.power_w[j] = float(
                        self.cfg.power_min_w + sig[j] * (self.cfg.power_max_w - self.cfg.power_min_w)
                    )
                idx += pow_dim
            else:
                for j in range(K):
                    decision.power_w[j] = _power_default(cfg.comm)

            if optimize_freq:
                raw = vec[idx : idx + freq_dim]
                sig = _sigmoid(raw)
                decision.freq_uav_hz = float(
                    self.cfg.uav_freq_min_hz + sig[0] * (self.cfg.uav_freq_max_hz - self.cfg.uav_freq_min_hz)
                )
                for j in range(K):
                    decision.freq_md_hz[j] = float(
                        self.cfg.md_freq_min_hz + sig[j + 1] * (self.cfg.md_freq_max_hz - self.cfg.md_freq_min_hz)
                    )
            else:
                decision.freq_uav_hz = cfg.energy.uav_cpu_hz
                for j in range(K):
                    decision.freq_md_hz[j] = cfg.energy.md_cpu_hz

            return decision

        def evaluate(vec: np.ndarray) -> float:
            decision = decode(vec)
            arrival_rate = cfg.tasks.arrival_rate
            if cfg.tasks.arrival_rate_mode == "sampled":
                per_md = {j: 0 for j in range(len(md_positions))}
                for t in tasks:
                    per_md[t.md_id] = per_md.get(t.md_id, 0) + 1
                L = cfg.sim.decision_dt_s
                arrival_rate = {j: per_md[j] / max(1e-6, L) for j in per_md}
            metrics = compute_metrics(
                tasks=tasks,
                md_positions=md_positions,
                uav_pos=uav_pos,
                energy_cfg=cfg.energy,
                comm_cfg=cfg.comm,
                delay_cfg=cfg.delay,
                sim_cfg=cfg.sim,
                E_mov=E_mov,
                world=world,
                offload_cfg=cfg.offload,
                cloud_cfg=cfg.cloud,
                arrival_rate=arrival_rate,
                alloc=decision,
                offload_granularity=self.cfg.offload_granularity,
            )
            return metrics.D_total + cfg.sim.epsilon * metrics.E_total

        pbest = pos.copy()
        pbest_val = np.array([evaluate(pos[i]) for i in range(particles)], dtype=float)
        gbest_idx = int(np.argmin(pbest_val))
        gbest = pos[gbest_idx].copy()
        gbest_val = float(pbest_val[gbest_idx])
        no_improve = 0

        for _ in range(iters):
            for i in range(particles):
                val = evaluate(pos[i])
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = pos[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest = pos[i].copy()
                    no_improve = 0
            no_improve += 1

            if no_improve >= no_improve_limit or gbest_val <= tol:
                break

            r1 = self.rng.random((particles, dim))
            r2 = self.rng.random((particles, dim))
            vel = inertia * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            vel = np.clip(vel, -vel_clip, vel_clip)
            pos = pos + vel

        decision = decode(gbest)
        decision.meta.update({"status": "pso", "best_S": gbest_val})
        return decision
