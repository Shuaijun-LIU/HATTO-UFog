"""Fuzzy PID controller with heuristic gain scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ufog_network.control.pid import PIDController, PIDGains


@dataclass
class FuzzyConfig:
    labels: Tuple[str, ...] = ("NB", "NM", "NS", "ZO", "PS", "PM", "PB")
    centers: Tuple[float, ...] = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
    label_values_kp: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    label_values_ki: Tuple[float, ...] = (-1.5, -1.0, -1.0, 0.0, 1.0, 1.0, 1.5)
    label_values_kd: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    rule_table: Tuple[Tuple[str, ...], ...] = ()


class FuzzyPIDController(PIDController):
    def __init__(self, base_gains: PIDGains, fuzzy: FuzzyConfig | None = None, integral_limit: float | None = None) -> None:
        self.base_gains = base_gains
        self.fuzzy = fuzzy or FuzzyConfig()
        super().__init__(gains=base_gains, integral_limit=integral_limit)

    def _membership(self, x: float) -> Tuple[float, ...]:
        # Triangular membership across centers
        centers = self.fuzzy.centers
        if len(centers) < 2:
            return (1.0,)
        width = abs(centers[1] - centers[0])
        mu = []
        for c in centers:
            val = max(0.0, 1.0 - abs(x - c) / max(1e-6, width))
            mu.append(val)
        return tuple(mu)

    def _label_map(self) -> Tuple[dict, dict, dict]:
        kp_map = {lab: val for lab, val in zip(self.fuzzy.labels, self.fuzzy.label_values_kp)}
        ki_map = {lab: val for lab, val in zip(self.fuzzy.labels, self.fuzzy.label_values_ki)}
        kd_map = {lab: val for lab, val in zip(self.fuzzy.labels, self.fuzzy.label_values_kd)}
        return kp_map, ki_map, kd_map

    def _parse_rule(self, cell: str) -> Tuple[float, float, float]:
        kp_map, ki_map, kd_map = self._label_map()
        parts = cell.split("/")
        if len(parts) != 3:
            return 0.0, 0.0, 0.0
        kp = kp_map.get(parts[0].strip(), 0.0)
        ki = ki_map.get(parts[1].strip(), 0.0)
        kd = kd_map.get(parts[2].strip(), 0.0)
        return kp, ki, kd

    def _rule_delta(self, error: float, derror: float) -> Tuple[float, float, float]:
        e_m = self._membership(error)
        de_m = self._membership(derror)
        if not self.fuzzy.rule_table:
            return 0.0, 0.0, 0.0
        numerator_kp = 0.0
        numerator_ki = 0.0
        numerator_kd = 0.0
        denom = 0.0
        for i in range(len(self.fuzzy.rule_table)):
            for j in range(len(self.fuzzy.rule_table[i])):
                w = e_m[i] * de_m[j]
                dk = self._parse_rule(self.fuzzy.rule_table[i][j])
                numerator_kp += w * dk[0]
                numerator_ki += w * dk[1]
                numerator_kd += w * dk[2]
                denom += w
        denom = max(1e-6, denom)
        return numerator_kp / denom, numerator_ki / denom, numerator_kd / denom

    def _scaled_gains(self, error: float, derror: float) -> PIDGains:
        dk_p, dk_i, dk_d = self._rule_delta(error, derror)
        return PIDGains(
            kp=self.base_gains.kp + dk_p,
            ki=self.base_gains.ki + dk_i,
            kd=self.base_gains.kd + dk_d,
        )

    def step(self, target: float, current: float, dt: float) -> float:
        error = target - current
        derror = error - self.prev_error if self.initialized else 0.0
        self.gains = self._scaled_gains(error, derror)
        return super().step(target, current, dt)
