"""Task 4 experiment: what happens with 15- or 30-minute smart-meter data instead of 1-minute data.

The 1-minute cleaned data are averaged into r-minute intervals (aggregate and washing-machine
power), the hurdle model is retrained on the coarse series with the same houses and splits, and
its interval predictions are spread evenly over the minutes of each interval. They are then
scored against the *same* 1-minute ground truth and eligible minutes as the 1-minute model, so
every metric is directly comparable. A coarse model cannot place power within an interval, which
the minute-level metrics deliberately expose, and the cycle rule's 300 W peak test is harder to pass
once power is averaged, so the resolution-neutral measures are daily energy and wash-day detection
(was there a wash that day: actual or predicted energy >= ``WASH_DAY_KWH``).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cycles import CycleParams
from .features import build_features
from .models import LGBMHurdle
from .plotting import BLUE, INK_2, MUTED, ORANGE, VIOLET, save, time_axis
from .metrics import daily_energy
from .stage_model import eligible_mask, evaluate, per_minute_frame, tune_threshold
from .stage_prepare import load_cached

log = logging.getLogger(__name__)
LOCAL_TZ = "Europe/London"
MIN_COVERAGE = 0.8  # an interval is used only if >= 80 % of its minutes are present / eligible
WASH_DAY_KWH = 0.1  # a day counts as a wash day above this washing-machine energy


def _samples(minutes: tuple[int, ...], r: int, floor: int, odd: bool) -> tuple[int, ...]:
    """Convert durations in minutes to sample counts at resolution r (deduplicated)."""
    out = []
    for m in minutes:
        n = max(floor, round(m / r))
        if odd and n % 2 == 0:
            n += 1
        out.append(n)
    return tuple(sorted(set(out)))


def feature_kwargs(r: int) -> dict[str, Any]:
    """Same feature durations as the 1-minute model, expressed in r-minute samples."""
    return {"windows": _samples((5, 15, 31, 61, 121, 241), r, 3, True),
            "offsets": _samples((1, 2, 5, 10, 20, 30, 60, 120), r, 1, False),
            "base_window": _samples((241,), r, 3, True)[0],
            "step_windows": _samples((31, 121), r, 3, True)}


def coarse_house(paths, house: int, r: int) -> dict[str, Any]:
    """r-minute aggregate, target and eligibility for one house, plus the 1-minute frame."""
    m = load_cached(paths, house)
    elig = eligible_mask(m)
    rule = f"{r}min"
    agg_ok = m["aggregate"].notna().resample(rule).mean() >= MIN_COVERAGE
    agg = m["aggregate"].resample(rule).mean().where(agg_ok)
    y_ok = elig.astype(float).resample(rule).mean() >= MIN_COVERAGE
    y = m["wm"].clip(0, 4000).where(elig).resample(rule).mean()
    return {"m": m, "elig": elig, "agg": agg, "y": y.astype("float32"), "y_ok": y_ok & agg_ok}


def stack(paths, houses: list[int], r: int) -> dict[str, Any]:
    Xs, ys = [], []
    for h in houses:
        c = coarse_house(paths, h, r)
        X = build_features(c["agg"], **feature_kwargs(r))
        keep = c["y_ok"].to_numpy()
        Xs.append(X[keep])
        ys.append(c["y"].to_numpy()[keep])
        log.info("House %d @ %d min: %s intervals", h, r, f"{keep.sum():,}")
    return {"X": pd.concat(Xs), "y": np.concatenate(ys)}


def to_minutes(pred_c: pd.Series, index: pd.DatetimeIndex, r: int) -> pd.Series:
    """Spread each interval's mean power evenly over its minutes (energy-preserving)."""
    return pred_c.reindex(index, method="ffill", limit=r - 1).fillna(0.0)


def wash_day_metrics(d: dict, pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Per house and pooled precision / recall of wash days (days with >= 90 % data)."""
    out, days = {}, []
    for h in np.unique(d["house"]):
        f = per_minute_frame(d, pred, h)
        e = daily_energy(f.fillna({"true": 0, "pred": 0}), ["true", "pred"])
        days.append(e)
        out[f"house_{h}"] = _day_scores(e)
    out["pooled"] = _day_scores(pd.concat(days))
    return out


def _day_scores(e: pd.DataFrame) -> dict[str, float]:
    t, p = e["true"] >= WASH_DAY_KWH, e["pred"] >= WASH_DAY_KWH
    tp = float((t & p).sum())
    prec = tp / p.sum() if p.sum() else np.nan
    rec = tp / t.sum() if t.sum() else np.nan
    return {"wash_day_precision": prec, "wash_day_recall": rec, "days": int(len(e)), "wash_days": int(t.sum())}


def score(d: dict, pred: np.ndarray, params: CycleParams, on_w: float, r: int) -> list[dict]:
    res = evaluate(d, pred, params, on_w)
    days = wash_day_metrics(d, pred)
    return [{"resolution_min": r, "scope": k, **v, **days[k]} for k, v in res.items()]


def run(cfg: dict[str, Any], te: dict[str, Any], pred_1min: np.ndarray) -> pd.DataFrame:
    """Train/evaluate the hurdle model at each configured resolution.

    ``te`` / ``pred_1min`` are the 1-minute test set and hurdle-model predictions, rescored here
    with the same functions as the reference rows.
    """
    paths, mc = cfg["_paths"], cfg["model"]
    params = CycleParams.from_config(cfg["cycle_detection"])
    on_w = mc["active_threshold_w"]
    rows = score(te, pred_1min, params, on_w, 1)
    always_off = evaluate(te, np.zeros(len(te["y"])), params, on_w)["pooled"]["daily_energy_mae_kwh"]
    example: dict[int, pd.Series] = {}
    for r in mc["resolutions_min"]:
        tr = stack(paths, mc["train_houses"], r)
        va = stack(paths, mc["val_houses"], r)
        model = LGBMHurdle(on_w, mc["random_state"])
        model.fit(tr["X"], tr["y"], X_val=va["X"], y_val=va["y"])
        model.threshold, _ = tune_threshold(model, va, on_w)
        log.info("%d-min hurdle threshold tuned on validation: %.2f", r, model.threshold)
        # score on the 1-minute ground truth of the test houses
        Xs, ys, ps, ts, hs = [], [], [], [], []
        for h in mc["test_houses"]:
            c = coarse_house(paths, h, r)
            X = build_features(c["agg"], **feature_kwargs(r))
            pred_c = pd.Series(model.predict(X), index=X.index).where(c["agg"].notna(), 0.0)
            pred = to_minutes(pred_c, c["m"].index, r)
            if h == mc["test_houses"][0]:
                example[r] = c["agg"]
            e = c["elig"].to_numpy()
            Xs.append(pd.DataFrame({"agg": c["m"]["aggregate"].to_numpy()[e]}))
            ys.append(c["m"]["wm"].clip(0, 4000).to_numpy()[e])
            ps.append(pred.to_numpy()[e])
            ts.append(c["m"].index[e])
            hs.append(np.full(e.sum(), h, dtype=np.int16))
        d = {"X": pd.concat(Xs, ignore_index=True), "y": np.concatenate(ys).astype("float32"),
             "ts": ts[0].append(ts[1:]) if len(ts) > 1 else ts[0], "house": np.concatenate(hs)}
        rows += score(d, np.concatenate(ps), params, on_w, r)
        del tr, va, d
    table = pd.DataFrame(rows).round(4)
    table.to_csv(paths.tables / "task3_resolution.csv", index=False)
    plot_resolution(table, always_off, paths.figures / "task3_resolution.png")
    plot_resolution_example(paths, mc["test_houses"][0], example, paths.figures / "task3_resolution_example.png")
    log.info("Resolution experiment (pooled test):\n%s",
             table[table["scope"] == "pooled"][["resolution_min", "f1", "energy_error_pct", "daily_energy_mae_kwh",
                                                "wash_day_recall", "wash_day_precision", "cycle_recall"]].to_string())
    return table


def plot_resolution(table: pd.DataFrame, always_off_daily: float, path) -> None:
    """Pooled test metrics per resolution, with naive references: always-off daily error and the
    wash-day precision of calling every day a wash day (the base rate)."""
    t = table[table["scope"] == "pooled"].set_index("resolution_min")
    base_rate = t["wash_days"].iloc[0] / t["days"].iloc[0]
    refs = {"daily_energy_mae_kwh": (always_off_daily, "always-off"), "wash_day_precision": (base_rate, "every day")}
    labels = [f"{r} min" for r in t.index]
    specs = [("daily_energy_mae_kwh", "Daily energy MAE (kWh)  lower=better"),
             ("wash_day_recall", "Wash days found (recall)  higher=better"),
             ("wash_day_precision", "Wash-day precision  higher=better"),
             ("f1", "Minute on/off F1  higher=better")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.0), sharey=True)
    for ax, (col, title) in zip(axes, specs):
        vals = t[col].to_numpy()
        ax.barh(labels, vals, color=BLUE, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3g}", va="center", fontsize=8, color=INK_2)
        note = f"dashed = {refs[col][1]} ({refs[col][0]:.2f})" if col in refs else ""
        ax.set_title(f"{title}\n{note}", fontsize=9)
        if col == "daily_energy_mae_kwh":
            ax.set_xlim(0, max(np.nanmax(vals), always_off_daily) * 1.3)
        else:
            ax.set_xlim(0, 1)
        if col in refs:
            v = refs[col][0]
            ax.axvline(v, color=INK_2, ls="--", lw=1)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Input resolution")
    fig.suptitle("Same hurdle model and split, trained on 1-, 15- and 30-minute data (unseen test houses)",
                 fontweight="bold", y=1.1)
    save(fig, path)


def plot_resolution_example(paths, house: int, coarse: dict[int, pd.Series], path) -> None:
    """One day with washing in the first test house: what the aggregate looks like at each resolution."""
    m = load_cached(paths, house)
    loc_day = m.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    on = (m["wm"] >= 20).groupby(loc_day).sum()
    cover = m["aggregate"].notna().groupby(loc_day).mean()
    day = on.where(cover > 0.99).idxmax()
    start = pd.Timestamp(day).tz_localize(LOCAL_TZ).tz_convert("UTC")
    end = start + pd.Timedelta(days=1)
    w = m.loc[start:end - pd.Timedelta(minutes=1)]
    fig, ax = plt.subplots(figsize=(12, 4))
    x = w.index.tz_convert(LOCAL_TZ).tz_localize(None)
    ax.plot(x, w["aggregate"], color=MUTED, lw=0.6, label="Aggregate, 1 min")
    for (r, s), colour in zip(sorted(coarse.items()), (ORANGE, VIOLET)):
        c = s.loc[start:end - pd.Timedelta(minutes=1)]
        ax.step(c.index.tz_convert(LOCAL_TZ).tz_localize(None), c.to_numpy(), where="post", color=colour, lw=1.4,
                label=f"Aggregate, {r}-min mean")
    ax.fill_between(x, w["wm"].fillna(0), color=BLUE, alpha=0.35, lw=0, label="Actual washing machine (1 min)")
    ax.set_ylabel("Power (W)")
    ax.set_ylim(0, None)
    ax.set_title(f"House {house}, {pd.Timestamp(day):%a %d %b %Y}: the wash signature at 1, 15 and 30 minutes")
    time_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, path)
