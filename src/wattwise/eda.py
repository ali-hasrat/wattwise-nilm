"""Task 2 computations: availability/quality summaries and usage patterns (no plotting here)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .raw_cleaning import find_gaps

LOCAL_TZ = "Europe/London"


def prepare_wm(m: pd.DataFrame) -> pd.Series:
    """WM power for cycle detection: missing minutes and frozen (stuck) values set to NaN."""
    return m["wm"].where(~m["wm_stuck"].astype(bool) & (m["n_rows"] > 0))


def availability_summary(house: int, m: pd.DataFrame, cycles: pd.DataFrame, standby_w: float = 5,
                         on_w: float = 20) -> dict:
    """Coverage and data-quality figures for one house (``m`` = 1-min frame from clean_data)."""
    observed = m["n_rows"] > 0
    gaps = find_gaps(~observed, min_len=1)
    wm = m.loc[observed & ~m["wm_stuck"], "wm"]
    acc = cycles[cycles["accepted"]]
    obs_days = observed.sum() / 1440
    agg_wh = m.loc[observed, "aggregate"].sum() / 60
    wm_wh = wm.sum() / 60
    return {
        "house": house,
        "first_utc": m.index.min(), "last_utc": m.index.max(),
        "span_days": round((m.index.max() - m.index.min()).total_seconds() / 86400, 1),
        "observed_days": round(obs_days, 1),
        "coverage_pct": round(100 * observed.mean(), 1),
        "missing_intervals_ge_10min": int((gaps["minutes"] >= 10).sum()),
        "missing_intervals_ge_1day": int((gaps["minutes"] >= 1440).sum()),
        "longest_gap_days": round(gaps["minutes"].max() / 1440, 1) if len(gaps) else 0.0,
        "issues_minutes_pct": round(100 * m.loc[observed, "issues"].astype(bool).mean(), 2),
        "wm_stuck_minutes": int(m["wm_stuck"].sum()),
        "wm_zero_pct": round(100 * (wm == 0).mean(), 1),
        "wm_standby_0_5w_pct": round(100 * ((wm > 0) & (wm <= standby_w)).mean(), 1),
        "wm_active_ge20w_pct": round(100 * (wm >= on_w).mean(), 2),
        "candidate_events": int(len(cycles)),
        "cycles_accepted": int(len(acc)),
        **{f"rejected_{r}": int((cycles["reject_reason"] == r).sum())
           for r in ("too_short", "low_energy", "low_peak", "too_long")},
        "cycles_per_week": round(len(acc) / obs_days * 7, 2) if obs_days else np.nan,
        "wm_share_of_household_energy_pct": round(100 * wm_wh / agg_wh, 2) if agg_wh else np.nan,
    }


def enrich_cycles(house: int, cycles: pd.DataFrame, wm: pd.Series, heating_w: float = 1000,
                  peak_hours: tuple[int, int] = (16, 19)) -> pd.DataFrame:
    """Add house, local start time, weekday, heating energy and a peak-hour flag to accepted cycles."""
    c = cycles[cycles["accepted"]].copy()
    c.insert(0, "house", house)
    local = c["start"].dt.tz_convert(LOCAL_TZ)
    c["start_local"] = local.dt.tz_localize(None)
    c["hour_local"] = local.dt.hour
    c["weekday"] = local.dt.day_of_week  # 0 = Monday
    c["is_weekend"] = c["weekday"] >= 5
    p0 = wm.fillna(0).to_numpy()
    pos = wm.index.get_indexer(c["start"]), wm.index.get_indexer(c["end"])
    c["heating_wh"] = [p0[s:e + 1][p0[s:e + 1] >= heating_w].sum() / 60 for s, e in zip(*pos)]
    c["heating_share"] = c["heating_wh"] / c["energy_wh"]
    c["starts_in_peak"] = c["hour_local"].between(peak_hours[0], peak_hours[1] - 1)
    return c


def usage_tables(cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share (%) of each house's cycles by local start hour and by weekday."""
    hour = pd.crosstab(cycles["house"], cycles["hour_local"], normalize="index").reindex(columns=range(24), fill_value=0) * 100
    day = pd.crosstab(cycles["house"], cycles["weekday"], normalize="index").reindex(columns=range(7), fill_value=0) * 100
    return hour.round(2), day.round(2)


def cycle_characteristics(cycles: pd.DataFrame) -> pd.DataFrame:
    """Median (and IQR) per house of the main cycle characteristics."""
    g = cycles.groupby("house")
    out = pd.DataFrame({
        "cycles": g.size(),
        "duration_min_median": g["duration_min"].median(),
        "duration_min_iqr": g["duration_min"].quantile(0.75) - g["duration_min"].quantile(0.25),
        "energy_wh_median": g["energy_wh"].median(),
        "energy_wh_iqr": g["energy_wh"].quantile(0.75) - g["energy_wh"].quantile(0.25),
        "peak_w_median": g["peak_w"].median(),
        "heating_share_median": g["heating_share"].median(),
        "weekend_share_pct": 100 * g["is_weekend"].mean(),
        "peak_16_19_share_pct": 100 * g["starts_in_peak"].mean(),
    })
    return out.round(2)
