"""Simple kinematic dynamics and movement energy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import numpy as np

from ufog_network.config import SimConfig


@dataclass
class KinematicState:
    pos: np.ndarray
    vel: np.ndarray


def move_towards(
    state: KinematicState,
    target: Tuple[float, float, float],
    dt: float,
    max_speed: float,
    max_accel: float,
) -> Tuple[KinematicState, float, float, bool]:
    """Move toward target with speed/accel limits.

    Returns (new_state, distance, speed, speed_violation).
    """
    pos = state.pos
    vel = state.vel
    tgt = np.array(target, dtype=float)
    direction = tgt - pos
    dist = float(np.linalg.norm(direction))
    if dist > 1e-6:
        direction = direction / dist
    desired_speed = min(max_speed, dist / max(1e-6, dt))
    desired_vel = direction * desired_speed
    dv = desired_vel - vel
    dv_norm = float(np.linalg.norm(dv))
    max_dv = max_accel * dt
    if dv_norm > max_dv:
        dv = dv / max(1e-6, dv_norm) * max_dv
    new_vel = vel + dv
    speed = float(np.linalg.norm(new_vel))
    speed_violation = speed > max_speed + 1e-3
    if speed > max_speed:
        new_vel = new_vel / max(1e-6, speed) * max_speed
        speed = max_speed
    new_pos = pos + new_vel * dt
    step_dist = float(np.linalg.norm(new_pos - pos))
    return KinematicState(pos=new_pos, vel=new_vel), step_dist, speed, speed_violation


def movement_energy(distance: float, speed: float, dt: float, cfg: SimConfig) -> float:
    """Compute movement energy in Joules using a simple power model."""
    power = cfg.hover_power_w + cfg.move_power_coeff * (speed ** 2)
    return power * dt
