"""Baseline registry and factories (ACS, ACS-DS, CPS-ACO, GA-SCA, TD3)."""
from __future__ import annotations

from typing import Dict

from ufog_network.baselines.acs import ACSBaseline
from ufog_network.baselines.acs_ds import ACSDSBaseline
from ufog_network.baselines.cps_aco import CPSACOBaseline
from ufog_network.baselines.gasca import GASCABaseline
try:
    from ufog_network.baselines.td3 import TD3Baseline
    _TD3_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TD3Baseline = None
    _TD3_AVAILABLE = False
from ufog_network.config import BaselineConfig


def make_baseline(cfg: BaselineConfig):
    name = cfg.name.lower()
    preset = (cfg.presets or {}).get(name, {})
    params: Dict = {**preset, **(cfg.params or {})}
    if name == "acs":
        return ACSBaseline(params)
    if name == "cps_aco":
        return CPSACOBaseline(params)
    if name == "acs_ds":
        return ACSDSBaseline(params)
    if name == "ga_sca":
        return GASCABaseline(params)
    if name == "td3":
        if not _TD3_AVAILABLE or TD3Baseline is None:
            raise ImportError("TD3 baseline requires torch; install torch to enable.")
        return TD3Baseline(params)
    raise ValueError(f"Unknown baseline: {cfg.name}")


__all__ = [
    "ACSBaseline",
    "ACSDSBaseline",
    "CPSACOBaseline",
    "GASCABaseline",
    "make_baseline",
]
if _TD3_AVAILABLE:
    __all__.append("TD3Baseline")
