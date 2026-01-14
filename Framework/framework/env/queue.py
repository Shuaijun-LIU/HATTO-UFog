"""Queue state for transmission and computation backlogs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class QueueState:
    """Tracks backlog for per-MD transmission and UAV computation queues."""

    md_tx_bits: Dict[int, float] = field(default_factory=dict)
    uav_comp_cycles: float = 0.0

    @staticmethod
    def init(md_count: int) -> "QueueState":
        return QueueState(md_tx_bits={j: 0.0 for j in range(md_count)}, uav_comp_cycles=0.0)

    def reset(self) -> None:
        for j in list(self.md_tx_bits.keys()):
            self.md_tx_bits[j] = 0.0
        self.uav_comp_cycles = 0.0
