"""TD3 training loop."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from ufog_network.baselines.td3.model import Actor, Critic
from ufog_network.baselines.td3.replay_buffer import ReplayBuffer
from ufog_network.config import Config
from ufog_network.seeding import make_rng


@dataclass
class TD3TrainConfig:
    steps: int = 20000
    start_steps: int = 1000
    update_after: int = 1000
    update_every: int = 50
    batch_size: int = 128
    gamma: float = 0.99
    polyak: float = 0.995
    act_noise: float = 0.1
    noise_clip: float = 0.3
    policy_delay: int = 2
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    buffer_size: int = 200000
    hidden: int = 256
    seed: int = 1234
    device: str = "cpu"


def _get_train_cfg(params: Dict) -> TD3TrainConfig:
    cfg = TD3TrainConfig()
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def train_td3(cfg: Config, output_dir: str, params: Dict | None = None) -> Dict:
    params = params or {}
    train_cfg = _get_train_cfg(params)
    rng = make_rng(train_cfg.seed)

    from ufog_network.sim.env import UAVEnv

    env = UAVEnv(cfg)
    obs = env.reset(seed=train_cfg.seed)
    obs_dim = obs.shape[0]
    act_dim = 3

    device = torch.device(train_cfg.device)
    actor = Actor(obs_dim, act_dim, hidden=train_cfg.hidden).to(device)
    actor_t = Actor(obs_dim, act_dim, hidden=train_cfg.hidden).to(device)
    critic = Critic(obs_dim, act_dim, hidden=train_cfg.hidden).to(device)
    critic_t = Critic(obs_dim, act_dim, hidden=train_cfg.hidden).to(device)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=train_cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=train_cfg.critic_lr)

    replay = ReplayBuffer.create(obs_dim, act_dim, train_cfg.buffer_size)

    total_reward = 0.0
    for t in range(train_cfg.steps):
        if t < train_cfg.start_steps:
            act = rng.uniform(-1.0, 1.0, size=act_dim)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act = actor(obs_t).cpu().numpy()[0]
                act += train_cfg.act_noise * rng.normal(size=act_dim)
                act = np.clip(act, -1.0, 1.0)

        next_obs, reward, done, _info = env.step(act)
        replay.store(obs, act, reward, next_obs, float(done))
        obs = next_obs
        total_reward += reward

        if done:
            obs = env.reset(seed=train_cfg.seed + t)

        if t >= train_cfg.update_after and t % train_cfg.update_every == 0:
            for _ in range(train_cfg.update_every):
                batch = replay.sample_batch(train_cfg.batch_size, device=str(device))
                with torch.no_grad():
                    noise = (torch.randn_like(batch["act"]) * train_cfg.act_noise).clamp(
                        -train_cfg.noise_clip, train_cfg.noise_clip
                    )
                    act_t = (actor_t(batch["obs2"]) + noise).clamp(-1.0, 1.0)
                    q1_t, q2_t = critic_t(batch["obs2"], act_t)
                    q_t = torch.min(q1_t, q2_t)
                    target = batch["rew"] + train_cfg.gamma * (1.0 - batch["done"]) * q_t

                q1, q2 = critic(batch["obs"], batch["act"])
                critic_loss = ((q1 - target).pow(2).mean() + (q2 - target).pow(2).mean())
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                if t % train_cfg.policy_delay == 0:
                    actor_loss = -critic.q1_only(batch["obs"], actor(batch["obs"])).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # Polyak averaging
                    with torch.no_grad():
                        for p, p_t in zip(actor.parameters(), actor_t.parameters()):
                            p_t.data.mul_(train_cfg.polyak)
                            p_t.data.add_((1 - train_cfg.polyak) * p.data)
                        for p, p_t in zip(critic.parameters(), critic_t.parameters()):
                            p_t.data.mul_(train_cfg.polyak)
                            p_t.data.add_((1 - train_cfg.polyak) * p.data)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actor_path = output / "td3_actor.pt"
    critic_path = output / "td3_critic.pt"
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)

    summary = {
        "steps": train_cfg.steps,
        "total_reward": total_reward,
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
    }
    (output / "td3_train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
