"""Heuristic resource allocator for stable, reproducible runs.

This allocator is intentionally lightweight:
- Select a small set of "served" MDs based on distance (and optional LoS feasibility),
- Offload served MD tasks to the UAV, others to local MD compute,
- Provide consistent power/frequency/channel allocations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math

from ufog_network.alloc.base import AllocationDecision, ResourceAllocator
from ufog_network.env.metrics import gamma_channel_allocation, uniform_channel_allocation, round_channels


class HeuristicAllocator(ResourceAllocator):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def allocate(
        self,
        tasks: List[Any],
        md_positions: List[Tuple[float, float, float]],
        uav_pos: Tuple[float, float, float],
        cfg: Any,
        world: Any | None = None,
        E_mov: float = 0.0,
    ) -> AllocationDecision | None:
        if not md_positions:
            return AllocationDecision(meta={"status": "no_md"})
        if not tasks:
            return AllocationDecision(meta={"status": "no_tasks"})

        K = len(md_positions)
        size_sum: Dict[int, float] = {}
        for t in tasks:
            size_sum[t.md_id] = size_sum.get(t.md_id, 0.0) + float(t.size_bits)

        # Channels (provide to metrics; fractional by default)
        if cfg.comm.channel_mode == "uniform":
            channels_raw = uniform_channel_allocation(size_sum, cfg.comm)
        else:
            channels_raw = gamma_channel_allocation(size_sum, cfg.comm)
        channels = round_channels(channels_raw, cfg.comm)

        # Compute distances and optional blocking for MDs with tasks
        candidates: List[Tuple[float, int, bool]] = []
        for j in size_sum.keys():
            md_x, md_y, md_z = md_positions[j]
            dx = uav_pos[0] - md_x
            dy = uav_pos[1] - md_y
            dz = uav_pos[2] - md_z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            blocked = False
            if world is not None and cfg.comm.enable_los:
                blocked = not world.segment_is_free(uav_pos, (md_x, md_y, md_z), step=world.cfg.connect_step_m)
            candidates.append((dist, j, blocked))
        candidates.sort(key=lambda x: x[0])

        max_served = max(0, int(getattr(cfg.resource, "heuristic_max_served_mds", 1) or 0))
        max_dist = float(getattr(cfg.resource, "heuristic_max_distance_m", 0.0) or 0.0)
        force_nearest = bool(getattr(cfg.resource, "heuristic_force_nearest", True))
        frac_uav_default = float(getattr(cfg.resource, "heuristic_uav_fraction", 1.0))
        frac_uav_default = max(0.0, min(1.0, frac_uav_default))

        served: List[int] = []
        for dist, j, blocked in candidates:
            if len(served) >= max_served:
                break
            if getattr(cfg.offload, "heuristic_blocked_to_md", True) and blocked:
                continue
            if max_dist > 0.0 and dist > max_dist:
                continue
            served.append(j)
        if not served and force_nearest and candidates and max_served > 0:
            # Fall back to the nearest non-blocked MD if possible.
            for _dist, j, blocked in candidates:
                if getattr(cfg.offload, "heuristic_blocked_to_md", True) and blocked:
                    continue
                served = [j]
                break

        decision = AllocationDecision(meta={"status": "ok", "served_mds": served})
        decision.channel_alloc = channels

        # Power policy: keep it simple and consistent.
        if cfg.comm.power_mode == "fixed":
            p_default = float(cfg.comm.p_fixed_w)
        else:
            p_default = float(cfg.comm.p_max_w)
        p_default = max(float(cfg.comm.p_min_w), min(float(cfg.comm.p_max_w), p_default))
        for j in range(K):
            decision.power_w[j] = p_default

        # Frequency policy: use configured CPU values (paper-scale knobs remain in configs).
        decision.freq_uav_hz = float(cfg.energy.uav_cpu_hz)
        for j in range(K):
            decision.freq_md_hz[j] = float(cfg.energy.md_cpu_hz)

        # Offload: served MDs -> UAV, others -> MD; allow DC if explicitly enabled and size is large.
        for j in range(K):
            decision.offload_uav[j] = 0.0
            decision.offload_md[j] = 1.0
            decision.offload_dc[j] = 0.0

        if cfg.cloud is not None and cfg.cloud.enabled:
            dc_thresh = float(getattr(cfg.offload, "heuristic_dc_size_bits", 2.0e7))
        else:
            dc_thresh = float("inf")

        for j in size_sum.keys():
            if size_sum[j] >= dc_thresh:
                decision.offload_uav[j] = 0.0
                decision.offload_md[j] = 0.0
                decision.offload_dc[j] = 1.0
                continue
            if j in served:
                decision.offload_uav[j] = frac_uav_default
                decision.offload_md[j] = 1.0 - frac_uav_default
                decision.offload_dc[j] = 0.0
            else:
                decision.offload_uav[j] = 0.0
                decision.offload_md[j] = 1.0
                decision.offload_dc[j] = 0.0

        return decision


__all__ = ["HeuristicAllocator"]

