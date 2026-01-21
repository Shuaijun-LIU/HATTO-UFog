"""Energy model utilities for motor-based movement power."""
from __future__ import annotations

import math
from typing import Iterable

from ufog_network.config import DynamicsConfig, EnergyConfig


def _omega_bounds_rad_s(dyn_cfg: DynamicsConfig) -> tuple[float, float]:
    if dyn_cfg.omega_unit == "rpm":
        factor = 2.0 * math.pi / 60.0
        return dyn_cfg.omega_min * factor, dyn_cfg.omega_max * factor
    return dyn_cfg.omega_min, dyn_cfg.omega_max


def motor_power_w(omega: float, energy_cfg: EnergyConfig) -> float:
    """Approximate motor electrical power from rotor speed.

    omega: rad/s
    Uses back-EMF model: V = I*R + k_e*omega
    with k_e derived from kv (rpm/V).
    """
    # Convert omega to rpm
    rpm = omega * 60.0 / (2.0 * math.pi)
    kv = max(1e-6, energy_cfg.motor_kv)
    back_emf = rpm / kv  # volts
    voltage = energy_cfg.motor_voltage_v
    current = max(0.0, (voltage - back_emf) / max(1e-6, energy_cfg.motor_resistance_ohm))
    return voltage * current


def movement_energy_from_omegas(
    omegas: Iterable[float],
    dt: float,
    energy_cfg: EnergyConfig,
    dyn_cfg: DynamicsConfig,
) -> float:
    total_power = 0.0
    omega_min, omega_max = _omega_bounds_rad_s(dyn_cfg)
    for omega in omegas:
        omega_clamped = max(omega_min, min(omega_max, float(omega)))
        total_power += motor_power_w(omega_clamped, energy_cfg)
    return total_power * dt
