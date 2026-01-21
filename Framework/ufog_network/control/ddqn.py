"""DDQN gain adjuster for FEAR-PID."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import json
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def _expand_action_deltas(
    explicit: Sequence[Sequence[float]] | None,
    grid_kp: Sequence[float],
    grid_ki: Sequence[float],
    grid_kd: Sequence[float],
) -> List[Tuple[float, float, float]]:
    if explicit:
        return [tuple(map(float, row)) for row in explicit]
    if not grid_kp or not grid_ki or not grid_kd:
        return []
    return [(float(kp), float(ki), float(kd)) for kp in grid_kp for ki in grid_ki for kd in grid_kd]


@dataclass
class DDQNConfig:
    episodes: int = 200
    steps_per_episode: int = 200
    gamma: float = 0.98
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update: int = 10
    batch_size: int = 64
    buffer_size: int = 20000
    hidden: int = 64
    seed: int = 1234
    device: str = "cpu"
    scales: Tuple[float, ...] = (0.7, 0.9, 1.0, 1.1, 1.3)
    action_grid_kp: Tuple[float, ...] = (-0.2, 0.0, 0.2)
    action_grid_ki: Tuple[float, ...] = (-0.05, 0.0, 0.05)
    action_grid_kd: Tuple[float, ...] = (-0.1, 0.0, 0.1)
    action_deltas: List[List[float]] = field(default_factory=list)
    base_kp: float = 1.2
    base_ki: float = 0.0
    base_kd: float = 0.8
    kp_min: float = 0.0
    kp_max: float = 10.0
    ki_min: float = 0.0
    ki_max: float = 5.0
    kd_min: float = 0.0
    kd_max: float = 5.0
    integral_limit: float = 5.0
    u_clip: float = 50.0
    mass: float = 1.0
    error_limit: float = 5.0
    reward_w_e: float = 1.0
    reward_w_de: float = 0.4
    reward_w_du: float = 0.02
    reward_w_stable: float = 1.0
    axes: int = 3
    env_mode: str = "attitude"
    target_mode: str = "fixed"
    target_min: float = -1.0
    target_max: float = 1.0
    target_change_interval: int = 50
    disturbance_std: float = 0.0
    measurement_noise_std: float = 0.0
    inertia: Tuple[float, ...] = (0.02, 0.02, 0.04)
    angular_damping: float = 0.1
    angle_limit_rad: float = 0.7
    omega_limit: float = 4.0
    dt: float = 0.05
    damping: float = 0.2
    kp: float = 1.0
    kd: float = 0.2
    target: float = 1.0


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, 1), dtype=np.int64)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.next_state[self.ptr] = s2
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idx], device=device),
            torch.as_tensor(self.action[idx], device=device),
            torch.as_tensor(self.reward[idx], device=device),
            torch.as_tensor(self.next_state[idx], device=device),
            torch.as_tensor(self.done[idx], device=device),
        )


class DDQNGainAdjuster:
    def __init__(
        self,
        model_path: str,
        scales: Sequence[float] = (0.7, 0.9, 1.0, 1.1, 1.3),
        action_deltas: Sequence[Sequence[float]] | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.scales = tuple(scales)
        if action_deltas is None:
            summary_path = Path(model_path).with_name("fear_ddqn_summary.json")
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    action_deltas = summary.get("action_deltas")
                except json.JSONDecodeError:
                    action_deltas = None
        self.action_deltas = [tuple(map(float, row)) for row in action_deltas] if action_deltas else []
        self.use_delta = bool(self.action_deltas)
        state_dim = 5 if self.use_delta else 2
        action_dim = len(self.action_deltas) if self.use_delta else len(self.scales)
        self.model = QNetwork(state_dim=state_dim, action_dim=action_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.prev_error = 0.0
        self.current_delta = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.prev_error = 0.0
        self.current_delta = (0.0, 0.0, 0.0)

    def __call__(self, error: float) -> float | Tuple[float, float, float]:
        derror = error - self.prev_error
        self.prev_error = error
        if self.use_delta:
            state = torch.tensor([[error, derror, *self.current_delta]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q = self.model(state)
            action = int(torch.argmax(q, dim=1).item())
            delta = self.action_deltas[action]
            self.current_delta = delta
            return delta
        state = torch.tensor([[error, derror]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.model(state)
        action = int(torch.argmax(q, dim=1).item())
        return float(self.scales[action])


def train_ddqn_gain_adjuster(output_dir: str, cfg: DDQNConfig | None = None) -> dict:
    cfg = cfg or DDQNConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    action_deltas = _expand_action_deltas(cfg.action_deltas, cfg.action_grid_kp, cfg.action_grid_ki, cfg.action_grid_kd)
    use_delta = bool(action_deltas)
    state_dim = 5 if use_delta else 2
    action_dim = len(action_deltas) if use_delta else len(cfg.scales)

    device = torch.device(cfg.device)
    q = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden=cfg.hidden).to(device)
    q_t = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden=cfg.hidden).to(device)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size, state_dim=state_dim)

    epsilon = cfg.epsilon_start
    total_reward = 0.0

    axes = max(1, int(cfg.axes))
    env_mode = cfg.env_mode
    inertia = np.array(cfg.inertia if cfg.inertia else [0.02] * axes, dtype=np.float32)
    if inertia.size == 1:
        inertia = np.repeat(inertia, axes)
    elif inertia.size != axes:
        inertia = np.resize(inertia, axes)

    for ep in range(cfg.episodes):
        x = np.zeros(axes, dtype=np.float32)
        v = np.zeros(axes, dtype=np.float32)
        integral = np.zeros(axes, dtype=np.float32)
        if cfg.target_mode == "random":
            target = np.random.uniform(cfg.target_min, cfg.target_max, size=axes).astype(np.float32)
        else:
            target = np.ones(axes, dtype=np.float32) * cfg.target
        prev_error_meas = np.asarray(target - x, dtype=np.float32)
        prev_u = np.zeros(axes, dtype=np.float32)
        delta = (0.0, 0.0, 0.0)
        for t in range(cfg.steps_per_episode):
            if cfg.target_mode == "random" and cfg.target_change_interval > 0:
                if t % cfg.target_change_interval == 0 and t > 0:
                    target = np.random.uniform(cfg.target_min, cfg.target_max, size=axes).astype(np.float32)
            error = np.asarray(target - x, dtype=np.float32)
            noise = np.random.normal(0.0, cfg.measurement_noise_std, size=axes)
            error_meas = np.asarray(error + noise, dtype=np.float32)
            derror = np.asarray(error_meas - prev_error_meas, dtype=np.float32)
            prev_error_meas = error_meas
            E = float(np.asarray(error_meas).mean())
            EC = float(np.asarray(derror).mean())
            if use_delta:
                state = np.array([E, EC, *delta], dtype=np.float32)
            else:
                state = np.array([E, EC], dtype=np.float32)

            if np.random.rand() < epsilon:
                action = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    qvals = q(torch.tensor(state, device=device).unsqueeze(0))
                action = int(torch.argmax(qvals, dim=1).item())

            if use_delta:
                delta = action_deltas[action]
                kp = min(cfg.kp_max, max(cfg.kp_min, cfg.base_kp + delta[0]))
                ki = min(cfg.ki_max, max(cfg.ki_min, cfg.base_ki + delta[1]))
                kd = min(cfg.kd_max, max(cfg.kd_min, cfg.base_kd + delta[2]))
                integral = np.clip(integral + error_meas * cfg.dt, -cfg.integral_limit, cfg.integral_limit)
                u = kp * error_meas + ki * integral + kd * derror
            else:
                scale = cfg.scales[action]
                u = scale * (cfg.kp * error_meas + cfg.kd * derror)

            u = np.clip(u, -cfg.u_clip, cfg.u_clip)
            disturbance = np.random.normal(0.0, cfg.disturbance_std, size=axes)
            if env_mode == "attitude":
                gyro = np.cross(v, inertia * v)
                omega_dot = (u - cfg.angular_damping * v - gyro) / np.maximum(1e-6, inertia)
                v = v + omega_dot * cfg.dt + disturbance
                v = np.clip(v, -cfg.omega_limit, cfg.omega_limit)
                x = x + v * cfg.dt
                x = np.clip(x, -cfg.angle_limit_rad, cfg.angle_limit_rad)
            else:
                v = v + (u / max(1e-6, cfg.mass) - cfg.damping * v) * cfg.dt + disturbance
                x = x + v * cfg.dt
            next_error = np.asarray(target - x, dtype=np.float32)
            next_noise = np.random.normal(0.0, cfg.measurement_noise_std, size=axes)
            next_error_meas = np.asarray(next_error + next_noise, dtype=np.float32)
            next_derror = np.asarray(next_error_meas - error_meas, dtype=np.float32)

            reward = -(
                cfg.reward_w_e * float(np.asarray(np.abs(next_error)).mean())
                + cfg.reward_w_de * float(np.asarray(np.abs(next_derror)).mean())
                + cfg.reward_w_du * float(np.asarray((u - prev_u) ** 2).mean())
            )
            if np.max(np.abs(next_error)) > cfg.error_limit:
                reward -= cfg.reward_w_stable * float(np.max(np.abs(next_error)) - cfg.error_limit)
            prev_u = u
            done = 1.0 if t == cfg.steps_per_episode - 1 else 0.0

            next_E = float(np.asarray(next_error_meas).mean())
            next_EC = float(np.asarray(next_derror).mean())
            if use_delta:
                next_state = np.array([next_E, next_EC, *delta], dtype=np.float32)
            else:
                next_state = np.array([next_E, next_EC], dtype=np.float32)
            replay.store(state, action, reward, next_state, done)
            total_reward += reward

            if replay.size >= cfg.batch_size:
                s, a, r, s2, d = replay.sample(cfg.batch_size, device=device)
                with torch.no_grad():
                    next_actions = torch.argmax(q(s2), dim=1, keepdim=True)
                    q_next = q_t(s2).gather(1, next_actions)
                    target = r + cfg.gamma * (1.0 - d) * q_next
                q_pred = q(s).gather(1, a)
                loss = (q_pred - target).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        if (ep + 1) % cfg.target_update == 0:
            q_t.load_state_dict(q.state_dict())
        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "fear_ddqn.pt"
    torch.save(q.state_dict(), model_path)
    summary = {
        "episodes": cfg.episodes,
        "total_reward": total_reward,
        "model_path": str(model_path),
        "action_mode": "delta" if use_delta else "scale",
        "action_deltas": action_deltas if use_delta else None,
        "env_mode": env_mode,
    }
    (output / "fear_ddqn_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
