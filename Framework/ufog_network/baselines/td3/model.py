"""TD3 actor/critic models."""
from __future__ import annotations

import torch
from torch import nn


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim], activation=nn.ReLU, output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU, output_activation=nn.Identity)
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_only(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu)
