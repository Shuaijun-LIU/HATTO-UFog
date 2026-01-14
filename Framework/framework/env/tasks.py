"""Task generation and MD placement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from framework.config import TaskConfig
from framework.seeding import make_rng
from framework.env.world import World


@dataclass
class Task:
    md_id: int
    size_bits: float
    cycles: float  # CPU cycles per bit


def generate_md_positions(world: World, task_cfg: TaskConfig, seed: int) -> List[Tuple[float, float, float]]:
    rng = make_rng(seed)
    positions: List[Tuple[float, float, float]] = []
    md_count = task_cfg.md_count
    if task_cfg.md_distribution == "poisson":
        area = world.cfg.size_m * world.cfg.size_m
        md_count = int(rng.poisson(task_cfg.md_density * area))
        md_count = max(task_cfg.md_min_count, md_count)
    attempts = 0
    max_attempts = md_count * task_cfg.position_attempts_factor
    while len(positions) < md_count and attempts < max_attempts:
        attempts += 1
        x = (rng.random() - 0.5) * world.cfg.size_m
        y = (rng.random() - 0.5) * world.cfg.size_m
        z = world.terrain.height(x, y) + task_cfg.md_height_offset_m
        # Avoid buildings
        if any(b.contains((x, y, z + 1.0)) for b in world.buildings):
            continue
        positions.append((x, y, z))
    return positions


def sample_tasks(cfg: TaskConfig, md_count: int, seed: int) -> List[Task]:
    rng = make_rng(seed)
    tasks: List[Task] = []
    for j in range(md_count):
        # Poisson arrivals (mean arrival_rate)
        num = rng.poisson(cfg.arrival_rate)
        for _ in range(num):
            if cfg.task_size_dist == "exponential":
                size = rng.exponential(cfg.task_size_mean)
            else:
                size = rng.normal(cfg.task_size_mean, cfg.task_size_std)
            size = max(cfg.task_size_min_bits, min(cfg.task_size_max_bits, size))
            if cfg.cycles_dist == "exponential":
                cycles = rng.exponential(cfg.cycles_mean)
            else:
                cycles = rng.normal(cfg.cycles_mean, cfg.cycles_std)
            cycles = max(cfg.cycles_min, min(cfg.cycles_max, cycles))
            tasks.append(Task(md_id=j, size_bits=size, cycles=cycles))
    return tasks
