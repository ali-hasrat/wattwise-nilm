"""Evaluation metrics for appliance disaggregation.

* MAE / RMSE (W)            per-minute power error. Dominated by the many off minutes, so an
                            always-off model scores deceptively well - read with the others.
* MAE when on (W)           error restricted to minutes where the machine is truly active.
* Energy error (%)          (sum(pred) - sum(true)) / sum(true) over the period: what a bill
                            breakdown would get wrong. Signed, so over/under-estimates show.
* Daily energy MAE (kWh)    mean absolute error of per-day WM energy (days with >= 90% data).
* Precision / recall / F1   minute on/off state (power >= threshold). Always-off has recall 0.
* Cycle recall / precision  event level: share of true cycles overlapped by a predicted cycle,
                            and of predicted cycles overlapping a true one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cycles import CycleParams, detect_cycles


def regression_metrics(y: np.ndarray, p: np.ndarray, on_w: float = 20) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y and p must be non-empty and the same length")
    err = p - y
    on_true, on_pred = y >= on_w, p >= on_w
    tp = float(np.sum(on_true & on_pred))
    prec = tp / on_pred.sum() if on_pred.sum() else 0.0
    rec = tp / on_true.sum() if on_true.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    total = y.sum()
    return {
        "mae_w": float(np.mean(np.abs(err))),
        "rmse_w": float(np.sqrt(np.mean(err ** 2))),
        "mae_on_w": float(np.mean(np.abs(err[on_true]))) if on_true.any() else np.nan,
        "energy_error_pct": float(100 * (p.sum() - total) / total) if total else np.nan,
        "precision": prec, "recall": rec, "f1": f1,
    }


def daily_energy(frame: pd.DataFrame, cols: list[str], tz: str = "Europe/London", min_coverage: float = 0.9) -> pd.DataFrame:
    """Per local day energy (kWh) for ``cols`` of a 1-min frame with an ``eval`` mask column."""
    local_day = frame.index.tz_convert(tz).tz_localize(None).normalize()
    g = frame.groupby(local_day)
    out = g[cols].sum() / 60 / 1000
    out["coverage"] = g["eval"].mean()
    return out[out["coverage"] >= min_coverage]


def daily_energy_mae(frame: pd.DataFrame, true_col: str, pred_col: str) -> float:
    d = daily_energy(frame, [true_col, pred_col])
    return float((d[pred_col] - d[true_col]).abs().mean()) if len(d) else np.nan


def _overlaps(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """For each interval in a: does it overlap any interval in b."""
    if a.empty:
        return np.zeros(0, dtype=bool)
    if b.empty:
        return np.zeros(len(a), dtype=bool)
    bs, be = b["start"].to_numpy(), b["end"].to_numpy()
    order = np.argsort(bs)
    bs, be = bs[order], be[order]
    be_max = np.maximum.accumulate(be)
    out = []
    for s, e in zip(a["start"].to_numpy(), a["end"].to_numpy()):
        i = np.searchsorted(bs, e, side="right")  # b intervals starting <= e
        out.append(i > 0 and be_max[i - 1] >= s)
    return np.array(out, dtype=bool)


def cycle_metrics(true_w: pd.Series, pred_w: pd.Series, params: CycleParams) -> dict[str, float]:
    t = detect_cycles(true_w, params)
    p = detect_cycles(pred_w, params)
    t, p = t[t["accepted"]], p[p["accepted"]]
    hit_t, hit_p = _overlaps(t, p), _overlaps(p, t)
    return {
        "true_cycles": int(len(t)), "pred_cycles": int(len(p)),
        "cycle_recall": float(hit_t.mean()) if len(t) else np.nan,
        "cycle_precision": float(hit_p.mean()) if len(p) else np.nan,
    }
