"""Quadrotor mixer utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MixerConfig:
    kf: float = 1e-5
    km: float = 1e-6
    omega_min: float = 200.0
    omega_max: float = 5000.0
    arm_length: float = 0.25


def mix_thrust_torque(
    thrust: float, tau_x: float, tau_y: float, tau_z: float, cfg: MixerConfig
) -> Tuple[float, float, float, float]:
    """Compute rotor speeds from thrust/torques with a simple mixer."""
    # X configuration mixing with arm length and yaw torque scaling
    l = max(cfg.arm_length, 1e-6)
    km_ratio = cfg.kf / max(cfg.km, 1e-9)
    a = tau_x / l
    b = tau_y / l
    c = tau_z * km_ratio
    f1 = thrust / 4 - b / 2 + c / 4
    f2 = thrust / 4 - a / 2 - c / 4
    f3 = thrust / 4 + b / 2 + c / 4
    f4 = thrust / 4 + a / 2 - c / 4
    # Convert to omega using thrust coefficient
    def to_omega(f: float) -> float:
        f = max(0.0, f)
        omega = (f / max(cfg.kf, 1e-12)) ** 0.5
        return max(cfg.omega_min, min(cfg.omega_max, omega))

    return (to_omega(f1), to_omega(f2), to_omega(f3), to_omega(f4))
