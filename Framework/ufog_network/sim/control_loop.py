"""Low-level position + attitude control loop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import math
import numpy as np

from ufog_network.config import ControlConfig, DynamicsConfig, SimConfig
from ufog_network.control import PIDController, PIDGains, FuzzyPIDController, FEARPIDController, MixerConfig, mix_thrust_torque, DDQNGainAdjuster, FuzzyConfig
from ufog_network.sim.rigid_body import RigidBodyState, euler_from_quat


@dataclass
class ControlOutput:
    omegas: np.ndarray
    thrust: float
    tau: np.ndarray
    desired_att: Tuple[float, float, float]
    saturated: bool


def _expand_action_deltas(cfg: ControlConfig) -> Tuple[Tuple[float, float, float], ...]:
    if cfg.fear_action_deltas:
        return tuple(tuple(map(float, row)) for row in cfg.fear_action_deltas)
    if not cfg.fear_action_grid_kp or not cfg.fear_action_grid_ki or not cfg.fear_action_grid_kd:
        return ()
    return tuple(
        (float(kp), float(ki), float(kd))
        for kp in cfg.fear_action_grid_kp
        for ki in cfg.fear_action_grid_ki
        for kd in cfg.fear_action_grid_kd
    )


def _make_pid(
    kind: str,
    gains: PIDGains,
    model_path: str,
    scales: Tuple[float, ...],
    action_deltas: Tuple[Tuple[float, float, float], ...],
    fuzzy_cfg: FuzzyConfig,
    integral_limit: float | None,
) -> PIDController:
    if kind == "fuzzy":
        return FuzzyPIDController(base_gains=gains, fuzzy=fuzzy_cfg, integral_limit=integral_limit)
    if kind == "fear":
        if model_path:
            # Prefer model summary action space if available to avoid mismatches.
            summary_path = Path(model_path).with_name("fear_ddqn_summary.json")
            use_action_deltas = action_deltas if not summary_path.exists() else None
            adjuster = DDQNGainAdjuster(model_path, scales=scales, action_deltas=use_action_deltas)
        else:
            adjuster = None
        return FEARPIDController(
            base_gains=gains,
            gain_adjuster=adjuster,
            fuzzy=fuzzy_cfg,
            integral_limit=integral_limit,
        )
    return PIDController(gains=gains, integral_limit=integral_limit)


class LowLevelController:
    def __init__(self, control_cfg: ControlConfig, dynamics_cfg: DynamicsConfig, sim_cfg: SimConfig) -> None:
        self.control_cfg = control_cfg
        self.dynamics_cfg = dynamics_cfg
        self.sim_cfg = sim_cfg
        pos_gains = PIDGains(control_cfg.pos_kp, control_cfg.pos_ki, control_cfg.pos_kd)
        att_gains = PIDGains(control_cfg.att_kp, control_cfg.att_ki, control_cfg.att_kd)
        yaw_gains = PIDGains(control_cfg.yaw_kp, control_cfg.yaw_ki, control_cfg.yaw_kd)
        fuzzy_cfg = FuzzyConfig(
            labels=tuple(control_cfg.fuzzy_labels),
            centers=tuple(control_cfg.fuzzy_centers),
            label_values_kp=tuple(control_cfg.fuzzy_label_values_kp),
            label_values_ki=tuple(control_cfg.fuzzy_label_values_ki),
            label_values_kd=tuple(control_cfg.fuzzy_label_values_kd),
            rule_table=tuple(tuple(row) for row in control_cfg.fuzzy_rule_table),
        )
        scales = tuple(control_cfg.fear_scales)
        action_deltas = _expand_action_deltas(control_cfg)
        integral_limit = control_cfg.integral_limit
        if control_cfg.mode == "fear":
            pos_kind = "fuzzy"
            att_kind = "fear"
        else:
            pos_kind = control_cfg.mode
            att_kind = control_cfg.mode
        self.px = _make_pid(
            pos_kind,
            pos_gains,
            control_cfg.fear_model_path,
            scales,
            action_deltas,
            fuzzy_cfg,
            integral_limit,
        )
        self.py = _make_pid(
            pos_kind,
            pos_gains,
            control_cfg.fear_model_path,
            scales,
            action_deltas,
            fuzzy_cfg,
            integral_limit,
        )
        self.pz = _make_pid(
            pos_kind,
            pos_gains,
            control_cfg.fear_model_path,
            scales,
            action_deltas,
            fuzzy_cfg,
            integral_limit,
        )
        self.roll = _make_pid(
            att_kind,
            att_gains,
            control_cfg.fear_model_path,
            scales,
            action_deltas,
            fuzzy_cfg,
            integral_limit,
        )
        self.pitch = _make_pid(
            att_kind,
            att_gains,
            control_cfg.fear_model_path,
            scales,
            action_deltas,
            fuzzy_cfg,
            integral_limit,
        )
        self.yaw = _make_pid(
            att_kind,
            yaw_gains,
            control_cfg.fear_model_path,
            scales,
            action_deltas,
            fuzzy_cfg,
            integral_limit,
        )
        # Propagate FEAR gain limits if applicable
        for ctrl in (self.px, self.py, self.pz, self.roll, self.pitch, self.yaw):
            if hasattr(ctrl, "fear"):
                ctrl.fear.kp_min = control_cfg.fear_kp_min
                ctrl.fear.kp_max = control_cfg.fear_kp_max
                ctrl.fear.ki_min = control_cfg.fear_ki_min
                ctrl.fear.ki_max = control_cfg.fear_ki_max
                ctrl.fear.kd_min = control_cfg.fear_kd_min
                ctrl.fear.kd_max = control_cfg.fear_kd_max
        omega_min = dynamics_cfg.omega_min
        omega_max = dynamics_cfg.omega_max
        if dynamics_cfg.omega_unit == "rpm":
            omega_min = omega_min * 2.0 * math.pi / 60.0
            omega_max = omega_max * 2.0 * math.pi / 60.0
        self.mixer = MixerConfig(
            kf=dynamics_cfg.kf,
            km=dynamics_cfg.km,
            omega_min=omega_min,
            omega_max=omega_max,
            arm_length=dynamics_cfg.arm_length_m,
        )

    def reset(self) -> None:
        for ctrl in (self.px, self.py, self.pz, self.roll, self.pitch, self.yaw):
            ctrl.reset()

    def step(self, state: RigidBodyState, target_pos: Tuple[float, float, float], dt: float) -> ControlOutput:
        # Position control to desired acceleration
        pos = state.pos
        vel = state.vel
        damping = self.control_cfg.vel_damping
        ax = self.px.step(target_pos[0], pos[0], dt) - damping * vel[0]
        ay = self.py.step(target_pos[1], pos[1], dt) - damping * vel[1]
        az = self.pz.step(target_pos[2], pos[2], dt) - damping * vel[2]
        accel_limit = max(0.0, float(self.sim_cfg.max_accel_m_s2))
        if accel_limit > 0:
            accel = np.array([ax, ay, az], dtype=float)
            norm = float(np.linalg.norm(accel))
            if norm > accel_limit and norm > 1e-6:
                accel = accel * (accel_limit / norm)
                ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])
        # Desired thrust vector
        g = self.dynamics_cfg.gravity_m_s2
        thrust_vec = np.array([ax, ay, az + g], dtype=float) * self.dynamics_cfg.mass_kg
        thrust_mag = float(np.linalg.norm(thrust_vec))
        if thrust_mag < 1e-6:
            thrust_mag = 1e-6
        z_b = thrust_vec / thrust_mag
        # Desired roll/pitch with yaw=0
        pitch = np.arcsin(np.clip(z_b[0], -1.0, 1.0))
        roll = np.arctan2(-z_b[1], z_b[2])
        yaw = 0.0
        # Limit tilt
        max_tilt = np.deg2rad(self.control_cfg.max_tilt_deg)
        roll = float(np.clip(roll, -max_tilt, max_tilt))
        pitch = float(np.clip(pitch, -max_tilt, max_tilt))

        # Attitude control
        cur_roll, cur_pitch, cur_yaw = euler_from_quat(state.quat)
        tau_x = self.roll.step(roll, cur_roll, dt)
        tau_y = self.pitch.step(pitch, cur_pitch, dt)
        tau_z = self.yaw.step(yaw, cur_yaw, dt)

        omegas = np.array(mix_thrust_torque(thrust_mag, tau_x, tau_y, tau_z, self.mixer), dtype=float)
        sat = bool(
            (omegas <= self.mixer.omega_min + 1e-6).any()
            or (omegas >= self.mixer.omega_max - 1e-6).any()
        )
        return ControlOutput(
            omegas=omegas,
            thrust=thrust_mag,
            tau=np.array([tau_x, tau_y, tau_z]),
            desired_att=(roll, pitch, yaw),
            saturated=sat,
        )
