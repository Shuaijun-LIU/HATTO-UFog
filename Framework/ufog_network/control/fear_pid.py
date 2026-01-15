"""FEAR-PID controller: fuzzy PID with optional DDQN gain adjustment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ufog_network.control.fuzzy_pid import FuzzyConfig, FuzzyPIDController
from ufog_network.control.pid import PIDGains


@dataclass
class FEARConfig:
    gain_clip: float = 2.0
    kp_min: float = 0.0
    kp_max: float = 10.0
    ki_min: float = 0.0
    ki_max: float = 5.0
    kd_min: float = 0.0
    kd_max: float = 5.0


class FEARPIDController:
    def __init__(
        self,
        base_gains: PIDGains,
        fuzzy: FuzzyConfig | None = None,
        fear: FEARConfig | None = None,
        integral_limit: float | None = None,
        gain_adjuster: Callable[[float], float] | None = None,
    ) -> None:
        self.controller = FuzzyPIDController(base_gains=base_gains, fuzzy=fuzzy, integral_limit=integral_limit)
        self.fear = fear or FEARConfig()
        self.gain_adjuster = gain_adjuster

    def reset(self) -> None:
        self.controller.reset()
        if hasattr(self.gain_adjuster, "reset"):
            self.gain_adjuster.reset()

    def step(self, target: float, current: float, dt: float) -> float:
        error = target - current
        adjust = None
        if self.gain_adjuster is not None:
            adjust = self.gain_adjuster(error)
        if isinstance(adjust, (tuple, list)) and len(adjust) == 3:
            kp = min(self.fear.kp_max, max(self.fear.kp_min, self.controller.base_gains.kp + adjust[0]))
            ki = min(self.fear.ki_max, max(self.fear.ki_min, self.controller.base_gains.ki + adjust[1]))
            kd = min(self.fear.kd_max, max(self.fear.kd_min, self.controller.base_gains.kd + adjust[2]))
            self.controller.base_gains = PIDGains(kp=kp, ki=ki, kd=kd)
            output = self.controller.step(target, current, dt)
            return output
        scale = 1.0
        if adjust is not None:
            scale = float(adjust)
        scale = max(1.0 / self.fear.gain_clip, min(self.fear.gain_clip, scale))
        output = self.controller.step(target, current, dt)
        return output * scale
