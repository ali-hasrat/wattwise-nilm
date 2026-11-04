"""Audit and repair of the time base of the official cleaned REFIT files.

Finding (House 1, Task 1): raw REFIT timestamps are UK wall-clock time. The official
cleaned files were *meant* to be converted to UTC, but for some BST periods they still
carry wall-clock time (e.g. House 1 from 2014-10-01 onward). Hour-of-day analysis would
then be off by one hour for those days.

Method: for every BST day, correlate the official 1-min aggregate with the raw 1-min
aggregate (on its local clock) under two hypotheses:

* ``utc``   official(t) == raw_local(t + 1 h)   (correctly converted)
* ``local`` official(t) == raw_local(t)         (still wall-clock)

The better hypothesis wins if r >= ``min_r`` and beats the other by ``min_margin``.
Undecided BST days inherit the verdict of the nearest decided BST day (the time base
changes piecewise, at file/part boundaries). GMT days need no correction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def raw_local_aggregate(raw_paths: list[Path], aggregate_max_w: float = 20000) -> pd.Series:
    """1-min mean raw aggregate on the raw (local wall-clock) time base, all parts combined."""
    frames = []
    for p in raw_paths:
        d = pd.read_csv(p, usecols=["Unix", "Aggregate"], engine="pyarrow")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    agg = d["Aggregate"].astype("float32")
    agg = agg.where((agg > 0) & (agg <= aggregate_max_w))
    minute = pd.to_datetime(d["Unix"], unit="s").dt.floor("1min")
    return agg.groupby(minute).mean().sort_index()


def bst_days(index: pd.DatetimeIndex, tz: str = "Europe/London") -> pd.Series:
    """For each calendar day in a naive index: True if UK clocks are on BST at noon."""
    days = pd.DatetimeIndex(np.unique(index.normalize()))
    noon = (days + pd.Timedelta(hours=12)).tz_localize(tz)
    offset = noon.map(lambda t: t.utcoffset().total_seconds() / 3600)
    return pd.Series(np.asarray(offset) == 1.0, index=days)


def classify_days(official: pd.Series, raw_local: pd.Series, tz: str = "Europe/London",
                  min_r: float = 0.8, min_margin: float = 0.2) -> pd.DataFrame:
    """Per-day verdict on the official file's time base (see module docstring).

    ``official`` and ``raw_local`` are 1-min series with naive DatetimeIndex.
    """
    raw = raw_local.reindex(official.index)
    raw_plus1h = raw_local.reindex(official.index + pd.Timedelta(hours=1))
    raw_plus1h.index = official.index
    frame = pd.DataFrame({"off": official.to_numpy(), "local": raw.to_numpy(), "utc": raw_plus1h.to_numpy()},
                         index=official.index)
    day = frame.index.normalize()

    def corr(g: pd.DataFrame, col: str) -> float:
        ok = g[["off", col]].dropna()
        if len(ok) < 120 or ok["off"].std() == 0 or ok[col].std() == 0:
            return np.nan
        return float(ok["off"].corr(ok[col]))

    rows = []
    is_bst = bst_days(frame.index, tz)
    for d, g in frame.groupby(day):
        r_local, r_utc = corr(g, "local"), corr(g, "utc")
        if not is_bst.get(d, False):
            verdict = "gmt"
        elif np.nanmax([r_local, r_utc, -1]) >= min_r and abs(np.nan_to_num(r_utc) - np.nan_to_num(r_local)) >= min_margin:
            verdict = "utc" if np.nan_to_num(r_utc) > np.nan_to_num(r_local) else "local"
        else:
            verdict = "undetermined"
        rows.append({"day": d, "bst": bool(is_bst.get(d, False)), "r_utc": r_utc, "r_local": r_local, "verdict": verdict})
    out = pd.DataFrame(rows).set_index("day")
    # Undecided BST days inherit the nearest decided BST verdict.
    decided = out["verdict"].where(out["verdict"].isin(["utc", "local"]))
    inferred = decided[out["bst"]].ffill().bfill()
    out["final"] = out["verdict"]
    und = out["verdict"] == "undetermined"
    out.loc[und, "final"] = inferred.reindex(out.index)[und].fillna("utc")
    return out


def corrected_utc_index(official_ts: pd.Series, verdicts: pd.DataFrame) -> pd.Series:
    """UTC timestamps: subtract 1 h on days whose official time base is local BST."""
    day = official_ts.dt.normalize()
    is_local = day.map(verdicts["final"].eq("local")).fillna(False).astype(bool)
    fixed = official_ts - pd.to_timedelta(np.where(is_local, 3600, 0), unit="s")
    return fixed.dt.tz_localize("UTC")


def summarise(verdicts: pd.DataFrame) -> dict[str, int]:
    v = verdicts
    return {
        "days": int(len(v)),
        "bst_days": int(v["bst"].sum()),
        "bst_days_already_utc": int((v["verdict"] == "utc").sum()),
        "bst_days_local_corrected": int((v["final"] == "local").sum()),
        "bst_days_undetermined": int((v["verdict"] == "undetermined").sum()),
        "bst_days_undetermined_inferred_local": int(((v["verdict"] == "undetermined") & (v["final"] == "local")).sum()),
    }
