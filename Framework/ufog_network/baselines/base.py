"""Baseline adapter interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class Action:
    target: Tuple[float, float, float]
    info: Dict[str, Any]


class Baseline:
    name = "base"

    def reset(self, world, seed: int) -> None:
        self.world = world
        self.seed = seed

    def plan(self, state: Dict[str, Any]):
        raise NotImplementedError

    def act(self, state: Dict[str, Any]) -> Action:
        raise NotImplementedError
