"""Replay buffer for TD3."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ReplayBuffer:
    obs_buf: np.ndarray
    obs2_buf: np.ndarray
    act_buf: np.ndarray
    rew_buf: np.ndarray
    done_buf: np.ndarray
    ptr: int
    size: int
    max_size: int

    @classmethod
    def create(cls, obs_dim: int, act_dim: int, max_size: int) -> "ReplayBuffer":
        return cls(
            obs_buf=np.zeros((max_size, obs_dim), dtype=np.float32),
            obs2_buf=np.zeros((max_size, obs_dim), dtype=np.float32),
            act_buf=np.zeros((max_size, act_dim), dtype=np.float32),
            rew_buf=np.zeros((max_size, 1), dtype=np.float32),
            done_buf=np.zeros((max_size, 1), dtype=np.float32),
            ptr=0,
            size=0,
            max_size=max_size,
        )

    def store(self, obs, act, rew, next_obs, done) -> None:
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int, device: str) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=device),
            obs2=torch.as_tensor(self.obs2_buf[idxs], device=device),
            act=torch.as_tensor(self.act_buf[idxs], device=device),
            rew=torch.as_tensor(self.rew_buf[idxs], device=device),
            done=torch.as_tensor(self.done_buf[idxs], device=device),
        )
        return batch
