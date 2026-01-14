"""Resource allocation module: PSO, random, and allocation policies."""
from __future__ import annotations

from framework.alloc.base import AllocationDecision, ResourceAllocator
from framework.alloc.pso import PSOAllocator
from framework.alloc.random_alloc import RandomAllocator
from framework.config import ResourceConfig


def make_resource_allocator(cfg: ResourceConfig) -> ResourceAllocator | None:
    if cfg.mode == "pso":
        return PSOAllocator(cfg)
    if cfg.mode == "random":
        return RandomAllocator(cfg)
    return None


__all__ = ["AllocationDecision", "ResourceAllocator", "PSOAllocator", "RandomAllocator", "make_resource_allocator"]
