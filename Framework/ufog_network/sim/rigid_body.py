"""Rigid body quadrotor dynamics (6DoF)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ufog_network.config import DynamicsConfig


@dataclass
class RigidBodyState:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray  # w, x, y, z
    omega: np.ndarray  # body angular velocity


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / max(1e-9, np.linalg.norm(q))


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=float,
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_from_quat(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = q
    # roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


class RigidBodyModel:
    def __init__(self, cfg: DynamicsConfig) -> None:
        self.cfg = cfg
        self.inertia = np.array([cfg.inertia_xx, cfg.inertia_yy, cfg.inertia_zz], dtype=float)

    def step(
        self,
        state: RigidBodyState,
        omegas: np.ndarray,
        dt: float,
        wind_vel: np.ndarray | None = None,
        wind_accel_gain: float = 0.0,
    ) -> Tuple[RigidBodyState, float, np.ndarray]:
        # Rotor thrusts
        omega_sq = omegas * omegas
        f = self.cfg.kf * omega_sq
        thrust = float(np.sum(f))
        # Torques (X configuration)
        tau_x = self.cfg.arm_length_m * (f[3] - f[1])
        tau_y = self.cfg.arm_length_m * (f[2] - f[0])
        tau_z = self.cfg.km * (omega_sq[0] - omega_sq[1] + omega_sq[2] - omega_sq[3])
        tau = np.array([tau_x, tau_y, tau_z], dtype=float)

        # Translational dynamics
        R = quat_to_rot(state.quat)
        thrust_world = R @ np.array([0.0, 0.0, thrust])
        gravity = np.array([0.0, 0.0, -self.cfg.gravity_m_s2])
        accel = thrust_world / self.cfg.mass_kg + gravity - self.cfg.linear_drag * state.vel
        if wind_vel is not None and wind_accel_gain > 0.0:
            accel = accel + wind_accel_gain * (wind_vel - state.vel)
        vel = state.vel + accel * dt
        pos = state.pos + vel * dt

        # Rotational dynamics (body frame)
        omega = state.omega
        omega_dot = (tau - np.cross(omega, self.inertia * omega)) / self.inertia
        omega_dot -= self.cfg.angular_drag * omega
        omega = omega + omega_dot * dt
        # Quaternion integration
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
        qdot = 0.5 * quat_multiply(state.quat, omega_quat)
        quat = quat_normalize(state.quat + qdot * dt)

        new_state = RigidBodyState(pos=pos, vel=vel, quat=quat, omega=omega)
        return new_state, thrust, tau
