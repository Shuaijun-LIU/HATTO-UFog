"""Resource allocation module: PSO, random, and allocation policies."""
from __future__ import annotations

from ufog_network.alloc.base import AllocationDecision, ResourceAllocator
from ufog_network.alloc.heuristic_alloc import HeuristicAllocator
from ufog_network.alloc.pso import PSOAllocator
from ufog_network.alloc.random_alloc import RandomAllocator
from ufog_network.config import ResourceConfig


def make_resource_allocator(cfg: ResourceConfig) -> ResourceAllocator | None:
    if cfg.mode == "heuristic":
        return HeuristicAllocator(cfg)
    if cfg.mode == "pso":
        return PSOAllocator(cfg)
    if cfg.mode == "random":
        return RandomAllocator(cfg)
    return None


__all__ = ["AllocationDecision", "ResourceAllocator", "HeuristicAllocator", "PSOAllocator", "RandomAllocator", "make_resource_allocator"]
