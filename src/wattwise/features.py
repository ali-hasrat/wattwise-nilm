"""Features for washing-machine disaggregation, computed from the household aggregate only.

The washing-machine ground truth is never an input. By default windows are centred
(offline disaggregation of stored meter data may look at minutes either side of t,
as in sequence-to-point NILM); ``causal=True`` restricts every feature to minutes <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/London"


def _roll(s: pd.Series, w: int, causal: bool):
    return s.rolling(w, center=not causal, min_periods=1)


def build_features(agg: pd.Series, causal: bool = False, windows: tuple[int, ...] = (5, 15, 31, 61, 121),
                   offsets: tuple[int, ...] = (1, 2, 5, 10, 20, 30, 60), heater_w: float = 1500,
                   base_window: int = 241, step_windows: tuple[int, ...] = (31, 121)) -> pd.DataFrame:
    """Feature frame (float32) aligned with ``agg`` (regular UTC index, NaN = missing).

    Windows and offsets are in samples; the defaults are for 1-minute data."""
    if not isinstance(agg.index, pd.DatetimeIndex):
        raise TypeError("aggregate must have a DatetimeIndex")
    a = agg.astype("float64")
    f: dict[str, pd.Series] = {"agg": a}
    base = _roll(a, base_window, causal).min()                     # slowly varying base load (fridges, standby)
    f[f"base_{base_window}"] = base
    above = a - base
    f["above_base"] = above
    d1 = a.diff()
    f["diff_prev"] = d1
    if not causal:
        f["diff_next"] = a.shift(-1) - a
    for w in windows:
        r = _roll(a, w, causal)
        f[f"mean_{w}"] = r.mean()
        f[f"std_{w}"] = r.std()
        f[f"max_{w}"] = r.max()
        f[f"min_{w}"] = r.min()
        f[f"absdiff_sum_{w}"] = _roll(d1.abs(), w, causal).sum()
        f[f"frac_heater_{w}"] = _roll((above >= heater_w).astype(float).where(a.notna()), w, causal).mean()
        f[f"frac_mid_{w}"] = _roll(((above >= 30) & (above < 600)).astype(float).where(a.notna()), w, causal).mean()
        f[f"missing_{w}"] = _roll(a.isna().astype(float), w, causal).mean()
    steps_up = (d1 >= heater_w).astype(float)
    steps_down = (d1 <= -heater_w).astype(float)
    for w in step_windows:
        f[f"steps_up_{w}"] = _roll(steps_up, w, causal).sum()
        f[f"steps_down_{w}"] = _roll(steps_down, w, causal).sum()
    for k in offsets:
        f[f"lag_{k}"] = a.shift(k) - a
        if not causal:
            f[f"lead_{k}"] = a.shift(-k) - a
    local = agg.index.tz_convert(LOCAL_TZ) if agg.index.tz is not None else agg.index
    hour = local.hour + local.minute / 60
    f["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=agg.index)
    f["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=agg.index)
    f["is_weekend"] = pd.Series((local.day_of_week >= 5).astype(float), index=agg.index)
    return pd.DataFrame(f).astype("float32")
