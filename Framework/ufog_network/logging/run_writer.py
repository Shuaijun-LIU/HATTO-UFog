"""Run directory writer."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from ufog_network.config import Config


def make_run_id(prefix: str = "run") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def prepare_run_dir(base: str, run_id: str) -> Path:
    p = Path(base) / run_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "artifacts").mkdir(exist_ok=True)
    return p


def write_config(path: Path, cfg: Config) -> None:
    (path / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))


def write_summary(path: Path, summary: Dict) -> None:
    (path / "summary.json").write_text(json.dumps(summary, indent=2))
