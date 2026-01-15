"""Random resource allocation (for ablations)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ufog_network.alloc.base import AllocationDecision, ResourceAllocator
from ufog_network.env.metrics import gamma_channel_allocation, uniform_channel_allocation, round_channels
from ufog_network.seeding import make_rng


class RandomAllocator(ResourceAllocator):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.rng = make_rng(cfg.random_seed)

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

        if cfg.comm.channel_mode == "uniform":
            channels_raw = uniform_channel_allocation(size_sum, cfg.comm)
        else:
            channels_raw = gamma_channel_allocation(size_sum, cfg.comm)
        channels = round_channels(channels_raw, cfg.comm)

        decision = AllocationDecision(channel_alloc=channels)
        use_dc = cfg.cloud.enabled
        for j in range(K):
            if use_dc:
                choice = int(self.rng.integers(0, 3))
            else:
                choice = int(self.rng.integers(0, 2))
            decision.offload_uav[j] = 1.0 if choice == 0 else 0.0
            decision.offload_md[j] = 1.0 if choice == 1 else 0.0
            decision.offload_dc[j] = 1.0 if choice == 2 else 0.0

        # Power allocation (uniform)
        if self.cfg.random_power_mode == "max":
            power = cfg.comm.p_max_w
        elif self.cfg.random_power_mode == "mid":
            power = 0.5 * (cfg.comm.p_min_w + cfg.comm.p_max_w)
        else:
            power = cfg.comm.p_fixed_w if cfg.comm.power_mode == "fixed" else cfg.comm.p_max_w
        decision.power_w = {j: float(power) for j in range(K)}

        # Frequency allocation (uniform)
        if self.cfg.random_freq_mode == "mid":
            f_uav = 0.5 * (self.cfg.uav_freq_min_hz + self.cfg.uav_freq_max_hz)
            f_md = 0.5 * (self.cfg.md_freq_min_hz + self.cfg.md_freq_max_hz)
        else:
            f_uav = cfg.energy.uav_cpu_hz
            f_md = cfg.energy.md_cpu_hz
        decision.freq_uav_hz = float(f_uav)
        decision.freq_md_hz = {j: float(f_md) for j in range(K)}
        decision.meta["status"] = "random"
        return decision
