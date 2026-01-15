"""Lightweight RL environment wrapper for TD3 training."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ufog_network.baselines.base import Action
from ufog_network.config import Config
from ufog_network.sim.simulator import Simulator


class UAVEnv:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sim = Simulator(cfg)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.cfg.sim.seed = seed
        self.sim = Simulator(self.cfg)
        return self._obs()

    def _obs(self) -> np.ndarray:
        pos = self.sim.state.pos
        vel = self.sim.state.vel
        targets = self.sim._targets()
        target = targets[self.sim.target_idx] if targets else (0.0, 0.0, 0.0)
        battery_frac = max(0.0, self.sim.battery_wh / max(1e-6, self.cfg.energy.battery_wh))
        obs = np.array(
            [
                pos[0],
                pos[1],
                pos[2],
                vel[0],
                vel[1],
                vel[2],
                target[0],
                target[1],
                target[2],
                battery_frac,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        scale = self.cfg.sim.max_speed_m_s * self.cfg.sim.decision_dt_s
        target = self.sim.state.pos + action * scale
        ext_action = Action(target=(float(target[0]), float(target[1]), float(target[2])), info={})
        row, done = self.sim.step(external_action=ext_action)
        obs = self._obs()
        if not row:
            return obs, 0.0, True, {}
        reward = -row["S"] * self.cfg.sim.rl_reward_scale
        reward -= self.cfg.sim.rl_collision_penalty * (row["viol_obstacle"] + row["viol_alt"])  # penalty for unsafe moves
        return obs, reward, done, row
