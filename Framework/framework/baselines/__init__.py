"""Baseline registry and factories (ACS, ACS-DS, CPS-ACO, GA-SCA, TD3)."""
from __future__ import annotations

from typing import Dict

from framework.baselines.acs import ACSBaseline
from framework.baselines.acs_ds import ACSDSBaseline
from framework.baselines.cps_aco import CPSACOBaseline
from framework.baselines.gasca import GASCABaseline
from framework.baselines.td3 import TD3Baseline
from framework.config import BaselineConfig


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
        return TD3Baseline(params)
    raise ValueError(f"Unknown baseline: {cfg.name}")


__all__ = [
    "ACSBaseline",
    "ACSDSBaseline",
    "CPSACOBaseline",
    "GASCABaseline",
    "TD3Baseline",
    "make_baseline",
]
