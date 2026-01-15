"""Validate framework imports and basic world generation."""
from __future__ import annotations

from ufog_network.config import Config
from ufog_network.env.generators import generate_world
from ufog_network.baselines import make_baseline


def main() -> int:
    cfg = Config()
    world = generate_world(cfg.world)
    # Instantiate each baseline
    for name in ["acs", "acs_ds", "ga_sca", "cps_aco", "td3"]:
        cfg.baseline.name = name
        try:
            baseline = make_baseline(cfg.baseline)
        except ImportError as exc:
            print(f"Skipping baseline {name}: {exc}")
            continue
        baseline.reset(world, cfg.sim.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
