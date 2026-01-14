"""Parquet logger for timeseries output."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from framework.schemas import timeseries_columns


class ParquetLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.rows: List[Dict] = []

    def append(self, row: Dict) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        # Ensure all columns exist
        for col, _dtype in timeseries_columns():
            if col not in df.columns:
                df[col] = None
        df.to_parquet(self.path, index=False)
        self.rows = []
