"""TD3 baseline adapter, model definitions, and training entrypoint."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from ufog_network.baselines.base import Action, Baseline
from ufog_network.baselines.td3.model import Actor
from ufog_network.baselines.td3.trainer import train_td3, TD3TrainConfig
from ufog_network.seeding import make_rng


class TD3Baseline(Baseline):
    name = "td3"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def reset(self, world, seed: int) -> None:
        super().reset(world, seed)
        self.rng = make_rng(seed)
        self.device = torch.device(self.params.get("device", "cpu"))
        self.obs_dim = int(self.params.get("obs_dim", 10))
        self.act_dim = int(self.params.get("act_dim", 3))
        self.actor = Actor(self.obs_dim, self.act_dim, hidden=int(self.params.get("hidden", 256))).to(self.device)
        model_path = self.params.get("model_path")
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.actor.load_state_dict(state)
        self.actor.eval()

    def _obs_from_state(self, state: Dict[str, Any]) -> np.ndarray:
        pos = np.array(state["uav_pos"], dtype=np.float32)
        vel = np.array(state.get("uav_vel", (0.0, 0.0, 0.0)), dtype=np.float32)
        targets = state["targets"]
        target_idx = state.get("target_idx", 0)
        target = np.array(targets[target_idx] if targets else (0.0, 0.0, 0.0), dtype=np.float32)
        battery = float(state.get("battery_wh", 1.0))
        battery_frac = battery / max(1e-6, float(state.get("battery_wh_max", 1.0)))
        obs = np.concatenate([pos, vel, target, np.array([battery_frac], dtype=np.float32)])
        return obs

    def act(self, state: Dict[str, Any]) -> Action:
        obs = self._obs_from_state(state)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        noise = float(self.params.get("action_noise", 0.0))
        if noise > 0:
            action = action + self.rng.normal(0.0, noise, size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        step_scale = float(self.params.get("step_scale_m", 10.0))
        pos = np.array(state["uav_pos"], dtype=float)
        target = pos + action * step_scale

        # Clamp to world bounds
        half = self.world.cfg.size_m / 2.0
        target[0] = float(np.clip(target[0], -half, half))
        target[1] = float(np.clip(target[1], -half, half))
        target[2] = float(np.clip(target[2], self.world.terrain.clearance_m + 5.0, self.world.cfg.height_m))

        # Ensure target altitude clears local terrain, but respect ceiling
        min_z = self.world.terrain.height(target[0], target[1]) + self.world.terrain.clearance_m + 5.0
        if min_z > self.world.cfg.height_m:
            min_z = self.world.cfg.height_m
        target[2] = float(np.clip(target[2], min_z, self.world.cfg.height_m))

        if not self.world.is_free((target[0], target[1], target[2])):
            safe_z = self.world.terrain.height(target[0], target[1]) + self.world.terrain.clearance_m + 10.0
            target[2] = max(target[2], safe_z)

        return Action(target=(float(target[0]), float(target[1]), float(target[2])), info={"status": "ok"})

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action = self.act(state)
        return {"status": "ok", "target": action.target, "info": action.info}


__all__ = ["TD3Baseline", "train_td3", "TD3TrainConfig"]
