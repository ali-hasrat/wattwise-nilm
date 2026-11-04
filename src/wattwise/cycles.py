"""Washing-machine cycle detection on a regular 1-min power series.

Rule (plain language): a minute is *active* when the washing machine draws at least
``on_threshold_w``. Active stretches separated by short pauses (<= ``max_gap_minutes``,
e.g. soaking or drum rest) belong to the same cycle. A candidate is accepted as a wash
cycle if it lasts at least ``min_duration_minutes``, uses at least ``min_energy_wh``,
reaches at least ``min_peak_w`` (heating or spinning), and is not implausibly long
(``max_duration_minutes``, which catches stuck sensors / mislabelled loads).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CycleParams:
    on_threshold_w: float = 20
    max_gap_minutes: int = 10
    min_duration_minutes: int = 15
    min_energy_wh: float = 50
    min_peak_w: float = 300
    max_duration_minutes: int = 360

    @classmethod
    def from_config(cls, d: dict) -> "CycleParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


CYCLE_COLUMNS = ["start", "end", "duration_min", "active_min", "missing_min", "peak_w", "mean_active_w",
                 "energy_wh", "accepted", "reject_reason"]


def detect_cycles(power: pd.Series, params: CycleParams) -> pd.DataFrame:
    """Return one row per candidate cycle (accepted and rejected) with its statistics.

    ``power`` must have a regular 1-min DatetimeIndex; NaN means missing and counts as
    inactive. ``end`` is the start of the last active minute; ``duration_min`` spans
    first to last active minute inclusive. Energy integrates the 1-min means over the
    span (Wh = sum(W) / 60), treating missing minutes as 0 (reported in ``missing_min``).
    """
    if power.empty:
        return pd.DataFrame(columns=CYCLE_COLUMNS)
    idx = power.index
    if len(idx) > 1:
        step = pd.Series(idx[1:] - idx[:-1])
        if not (step == pd.Timedelta("1min")).all():
            raise ValueError("detect_cycles needs a regular 1-minute index (use .asfreq('1min'))")
    p = power.to_numpy(dtype="float64")
    active = np.nan_to_num(p, nan=0.0) >= params.on_threshold_w
    pos = np.flatnonzero(active)
    if len(pos) == 0:
        return pd.DataFrame(columns=CYCLE_COLUMNS)
    breaks = np.flatnonzero(np.diff(pos) - 1 > params.max_gap_minutes)
    starts = np.r_[pos[0], pos[breaks + 1]]
    ends = np.r_[pos[breaks], pos[-1]]

    p0 = np.nan_to_num(p, nan=0.0)
    csum = np.r_[0.0, np.cumsum(p0)]
    cact = np.r_[0, np.cumsum(active)]
    cmiss = np.r_[0, np.cumsum(np.isnan(p))]
    dur = ends - starts + 1
    energy = (csum[ends + 1] - csum[starts]) / 60.0
    n_act = cact[ends + 1] - cact[starts]
    act_sum = np.array([p0[s:e + 1][active[s:e + 1]].sum() for s, e in zip(starts, ends)])
    peak = np.array([p0[s:e + 1].max() for s, e in zip(starts, ends)])
    out = pd.DataFrame({
        "start": idx[starts], "end": idx[ends], "duration_min": dur, "active_min": n_act,
        "missing_min": cmiss[ends + 1] - cmiss[starts], "peak_w": peak,
        "mean_active_w": act_sum / np.maximum(n_act, 1), "energy_wh": energy,
    })
    reason = np.full(len(out), "", dtype=object)
    reason[out["duration_min"] < params.min_duration_minutes] = "too_short"
    reason[(reason == "") & (out["energy_wh"] < params.min_energy_wh)] = "low_energy"
    reason[(reason == "") & (out["peak_w"] < params.min_peak_w)] = "low_peak"
    reason[(reason == "") & (out["duration_min"] > params.max_duration_minutes)] = "too_long"
    out["accepted"] = reason == ""
    out["reject_reason"] = reason
    return out


def cycle_mask(index: pd.DatetimeIndex, cycles: pd.DataFrame) -> pd.Series:
    """Boolean series: True for minutes inside an accepted cycle."""
    mask = pd.Series(False, index=index)
    for s, e in cycles.loc[cycles["accepted"], ["start", "end"]].itertuples(index=False):
        mask.loc[s:e] = True
    return mask
