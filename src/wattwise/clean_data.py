"""Loading the official cleaned REFIT data for the washing-machine houses at 1-min resolution."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import timebase
from .io import clean_house_path, raw_part_paths, read_clean_house

log = logging.getLogger(__name__)

MINUTE_COLUMNS = ["aggregate", "wm", "issues", "n_rows", "wm_stuck"]


def flag_stuck(wm: pd.Series, min_minutes: int, min_w: float = 5.0) -> pd.Series:
    """Minutes inside runs where WM power is non-zero and exactly constant for >= ``min_minutes``.

    The official files forward-fill sensor dropouts, so a frozen non-zero value is a filled
    gap, not a real wash (real cycles fluctuate minute to minute).
    """
    v = wm.round(1)
    run = (v != v.shift()).cumsum()
    length = v.groupby(run).transform("size")
    return (length >= min_minutes) & (v > min_w)


def load_house_minutely(clean_dir: Path, raw_dir: Path | None, house: int, wm_col: str,
                        freq: str = "1min", stuck_minutes: int = 60, tz: str = "Europe/London"
                        ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Official cleaned data for one house as a regular 1-min UTC frame.

    Columns: aggregate (W), wm (W), issues (1 if any reading in the minute had Issues=1),
    n_rows (readings in the minute; 0 = missing minute), wm_stuck (frozen value flag).
    If ``raw_dir`` is given, the time base is audited and repaired (see ``timebase``).
    """
    df = read_clean_house(clean_house_path(clean_dir, house), ["Unix", "Aggregate", wm_col, "Issues"])
    official_ts = pd.to_datetime(df["Unix"], unit="s")
    verdicts = None
    if raw_dir is not None:
        off_min = df["Aggregate"].astype("float32").groupby(official_ts.dt.floor("1min")).mean()
        raw_min = timebase.raw_local_aggregate(raw_part_paths(raw_dir, house))
        verdicts = timebase.classify_days(off_min, raw_min, tz)
        ts = timebase.corrected_utc_index(official_ts, verdicts)
    else:
        ts = official_ts.dt.tz_localize("UTC")

    data = pd.DataFrame({"aggregate": df["Aggregate"].astype("float32").to_numpy(),
                         "wm": df[wm_col].astype("float32").to_numpy(),
                         "issues": df["Issues"].astype("float32").to_numpy()}, index=pd.DatetimeIndex(ts))
    data = data.sort_index(kind="stable")
    r = data.resample(freq)
    m = r[["aggregate", "wm"]].mean().astype("float32")
    m["issues"] = r["issues"].max().fillna(0).astype("int8")
    m["n_rows"] = r.size().astype("int32")
    m.loc[m["n_rows"] == 0, ["aggregate", "wm"]] = np.nan
    m["wm_stuck"] = flag_stuck(m["wm"], stuck_minutes)
    m.index.name = "ts_utc"
    return m, verdicts
