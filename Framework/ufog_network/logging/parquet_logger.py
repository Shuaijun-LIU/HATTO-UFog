"""Parquet logger for timeseries output."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ufog_network.schemas import timeseries_columns


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
        # Ensure all columns exist and enforce dtypes
        schema = timeseries_columns()
        for col, dtype in schema:
            if col not in df.columns:
                if dtype.startswith("float"):
                    df[col] = np.nan
                else:
                    df[col] = pd.NA
        # Reorder to schema
        df = df[[col for col, _dtype in schema]]
        # Cast types
        for col, dtype in schema:
            if dtype.startswith("float"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype.startswith("int"):
                series = pd.to_numeric(df[col], errors="coerce")
                if dtype == "int8":
                    df[col] = series.astype("Int8")
                else:
                    df[col] = series.astype("Int64")
            else:
                df[col] = df[col].astype("string")
        df.to_parquet(self.path, index=False)
        self.rows = []
