"""Readers for the REFIT raw and cleaned CSV files with schema validation.

The original files are only ever read, never modified.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CHANNELS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
RAW_COLUMNS = ["Time", "Unix", *CHANNELS]
CLEAN_COLUMNS = [*RAW_COLUMNS, "Issues"]


class DataValidationError(ValueError):
    """Raised when an input file does not match the expected REFIT schema."""


def raw_part_paths(raw_dir: Path, house: int) -> list[Path]:
    """Return the raw CSV parts for a house, ordered Part1, Part2, ...

    File naming is inconsistent in the release (RAW_House1_Part1 vs RAW_House_13_Part1),
    so both spellings are matched.
    """
    pat = re.compile(rf"^RAW_House_?{house}_Part(\d+)\.csv$", re.IGNORECASE)
    found = []
    for p in Path(raw_dir).glob("*.csv"):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise FileNotFoundError(f"No raw files for House {house} in {raw_dir}")
    return [p for _, p in sorted(found)]


def clean_house_path(clean_dir: Path, house: int) -> Path:
    p = Path(clean_dir) / f"CLEAN_House{house}.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Cleaned file for House {house} not found: {p}")
    return p


def _check_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")
    if df.empty:
        raise DataValidationError(f"{path.name}: file contains no data rows")


def read_raw_part(path: Path) -> pd.DataFrame:
    """Read one raw REFIT part. Power channels as float32 (W); 'NaN' strings become NaN."""
    path = Path(path)
    if path.stat().st_size == 0:
        raise DataValidationError(f"{path.name}: file is empty")
    df = pd.read_csv(path, dtype={c: "float32" for c in CHANNELS} | {"Time": "string"}, na_values=["NaN"])
    _check_columns(df, RAW_COLUMNS, path)
    df["Unix"] = pd.to_numeric(df["Unix"], errors="coerce")
    log.info("Read %s: %s rows", path.name, f"{len(df):,}")
    return df


def read_clean_house(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns of an official cleaned REFIT file (fast pyarrow engine)."""
    path = Path(path)
    if path.stat().st_size == 0:
        raise DataValidationError(f"{path.name}: file is empty")
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise DataValidationError(f"{path.name}: missing columns {missing}")
    df = pd.read_csv(path, usecols=columns, engine="pyarrow")
    _check_columns(df, columns, path)
    return df
