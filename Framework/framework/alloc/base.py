"""Resource allocation interfaces and decision container."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class AllocationDecision:
    """Per-slot resource allocation decisions.

    Dicts are keyed by MD index.
    """

    power_w: Dict[int, float] = field(default_factory=dict)
    freq_md_hz: Dict[int, float] = field(default_factory=dict)
    freq_uav_hz: float | None = None
    offload_uav: Dict[int, float] = field(default_factory=dict)
    offload_md: Dict[int, float] = field(default_factory=dict)
    offload_dc: Dict[int, float] = field(default_factory=dict)
    channel_alloc: Dict[int, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class ResourceAllocator:
    """Base interface for resource allocation policies."""

    def allocate(
        self,
        tasks: List[Any],
        md_positions: List[Tuple[float, float, float]],
        uav_pos: Tuple[float, float, float],
        cfg: Any,
        world: Any | None = None,
        E_mov: float = 0.0,
    ) -> AllocationDecision | None:
        raise NotImplementedError

