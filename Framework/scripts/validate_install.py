"""Validate framework imports and basic world generation."""
from __future__ import annotations

from framework.config import Config
from framework.env.generators import generate_world
from framework.baselines import make_baseline


def main() -> int:
    cfg = Config()
    world = generate_world(cfg.world)
    # Instantiate each baseline
    for name in ["acs", "acs_ds", "ga_sca", "cps_aco", "td3"]:
        cfg.baseline.name = name
        baseline = make_baseline(cfg.baseline)
        baseline.reset(world, cfg.sim.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
