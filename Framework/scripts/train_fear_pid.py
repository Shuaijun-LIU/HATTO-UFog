"""Train a DDQN gain adjuster for FEAR-PID."""
from __future__ import annotations

import argparse

from ufog_network.config import load_config
from ufog_network.control.ddqn import DDQNConfig, train_ddqn_gain_adjuster


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config yaml/json")
    parser.add_argument("--output", default="runs/fear_pid", help="Output directory")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        full_cfg = load_config(args.config)
        cfg = full_cfg.fear_ddqn
    else:
        cfg = DDQNConfig()
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.steps is not None:
        cfg.steps_per_episode = args.steps
    summary = train_ddqn_gain_adjuster(args.output, cfg)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
