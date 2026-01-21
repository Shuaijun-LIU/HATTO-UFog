from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParquetInfo:
    path: Path
    num_rows: int
    columns: List[str]


def parquet_info(path: str | Path) -> ParquetInfo:
    p = Path(path).expanduser().resolve()
    pf = pq.ParquetFile(p)
    return ParquetInfo(path=p, num_rows=int(pf.metadata.num_rows), columns=list(pf.schema_arrow.names))


def infer_framework_run_dir(timeseries_path: str | Path) -> Path:
    p = Path(timeseries_path).expanduser().resolve()
    if p.is_dir():
        return p
    return p.parent


def find_optional_artifacts(run_dir: str | Path) -> Dict[str, Optional[Path]]:
    d = Path(run_dir).expanduser().resolve()
    out: Dict[str, Optional[Path]] = {"run_dir": d, "timeseries": None, "config": None, "summary": None, "world": None}
    ts = d / "timeseries.parquet"
    out["timeseries"] = ts if ts.exists() else None
    for name in ["config.json", "summary.json", "world.json"]:
        p = d / name
        out[name.split(".")[0]] = p if p.exists() else None
    return out


def required_columns_present(columns: List[str], required: List[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in required if c not in columns]
    return (len(missing) == 0, missing)

