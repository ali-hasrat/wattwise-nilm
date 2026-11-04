"""Prepare stage: 1-min caches of the official cleaned data for every configured washing-machine house."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from . import timebase
from .clean_data import load_house_minutely

log = logging.getLogger(__name__)


def cache_path(paths, house: int):
    return paths.data / f"clean_house{house}_1min.parquet"


def load_cached(paths, house: int) -> pd.DataFrame:
    p = cache_path(paths, house)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing - run `python run_pipeline.py --stage prepare` first")
    return pd.read_parquet(p)


def run(cfg: dict[str, Any]) -> pd.DataFrame:
    paths, wcfg = cfg["_paths"], cfg["washing_machine"]
    audit_raw = wcfg.get("audit_timebase", True)
    rows = []
    for house, col in wcfg["channels"].items():
        m, verdicts = load_house_minutely(paths.clean_dir, paths.raw_dir if audit_raw else None, int(house), col,
                                          wcfg["resample_freq"], wcfg["stuck_minutes"])
        m.to_parquet(cache_path(paths, int(house)))
        row = {"house": int(house), "wm_column": col, "minutes": len(m)}
        if verdicts is not None:
            verdicts.to_csv(paths.tables / f"timebase_days_house{house}.csv")
            row |= timebase.summarise(verdicts)
        rows.append(row)
        log.info("House %s cached: %s", house, row)
    audit = pd.DataFrame(rows)
    audit.to_csv(paths.tables / "timebase_audit.csv", index=False)
    return audit
