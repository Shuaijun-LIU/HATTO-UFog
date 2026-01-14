"""PID controller for scalar signals."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class PIDController:
    def __init__(self, gains: PIDGains, integral_limit: float | None = None) -> None:
        self.gains = gains
        self.integral_limit = integral_limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def step(self, target: float, current: float, dt: float) -> float:
        error = target - current
        if not self.initialized:
            self.prev_error = error
            self.initialized = True
        self.integral += error * dt
        if self.integral_limit is not None:
            lim = abs(self.integral_limit)
            if self.integral > lim:
                self.integral = lim
            elif self.integral < -lim:
                self.integral = -lim
        derivative = (error - self.prev_error) / max(1e-6, dt)
        self.prev_error = error
        return self.gains.kp * error + self.gains.ki * self.integral + self.gains.kd * derivative
