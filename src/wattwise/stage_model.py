"""Task 3 stage: train, select and evaluate washing-machine disaggregation models.

Validation design (household holdout):
* train houses   -> model fitting (off minutes sub-sampled with inverse-probability weights)
* validation     -> early stopping, hurdle threshold, model selection
* test houses    -> touched once, for the final report only
No house contributes rows to more than one split, so overlapping windows cannot leak.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cycles import CycleParams
from .features import build_features
from .metrics import cycle_metrics, daily_energy, daily_energy_mae, regression_metrics
from .models import AlwaysOff, ConstantMean, LGBMHurdle, LGBMRegression, UsageProfile
from .plotting import BLUE, GRID, INK_2, MUTED, ORANGE, RED, VIOLET, save, time_axis
from .stage_prepare import load_cached

log = logging.getLogger(__name__)
LOCAL_TZ = "Europe/London"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def eligible_mask(m: pd.DataFrame) -> pd.Series:
    """Minutes usable as labelled examples: observed, WM not stuck, no official Issues flag."""
    return (m["n_rows"] > 0) & m["aggregate"].notna() & m["wm"].notna() & ~m["wm_stuck"].astype(bool) & (m["issues"] == 0)


def house_frame(paths, house: int, causal: bool) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(features, target W, eligible mask) for one house on its full regular 1-min index."""
    m = load_cached(paths, house)
    X = build_features(m["aggregate"], causal=causal)
    y = m["wm"].clip(0, 4000).astype("float32")
    return X, y, eligible_mask(m)


def subsample_train(y: pd.Series, eligible: pd.Series, on_w: float, context: int, frac: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Keep every eligible minute within +/-``context`` of WM activity and a random ``frac`` of the
    rest, weighting the sampled rest by 1/frac so the training distribution stays unbiased."""
    active = (y >= on_w).astype(float)
    near = active.rolling(2 * context + 1, center=True, min_periods=1).max().to_numpy() > 0
    keep_far = rng.random(len(y)) < frac
    elig = eligible.to_numpy()
    keep = elig & (near | keep_far)
    weight = np.where(near, 1.0, 1.0 / frac)
    return keep, weight


def assemble(paths, houses: list[int], causal: bool, sample: dict | None = None, rng=None,
             time_slice: tuple[float, float] | None = None) -> dict[str, Any]:
    """Stack eligible rows of several houses. ``time_slice`` keeps a fraction of each house's period."""
    Xs, ys, ws, ts, hs = [], [], [], [], []
    for h in houses:
        X, y, elig = house_frame(paths, h, causal)
        if time_slice is not None:
            n = len(y)
            lo, hi = int(n * time_slice[0]), int(n * time_slice[1])
            sl = np.zeros(n, dtype=bool)
            sl[lo:hi] = True
            elig = elig & sl
        if sample is not None:
            keep, w = subsample_train(y, elig, sample["on_w"], sample["context"], sample["frac"], rng)
        else:
            keep, w = elig.to_numpy(), np.ones(len(y))
        Xs.append(X[keep])
        ys.append(y.to_numpy()[keep])
        ws.append(w[keep])
        ts.append(X.index[keep])
        hs.append(np.full(keep.sum(), h, dtype=np.int16))
        log.info("House %d: %s rows kept (%s eligible)", h, f"{keep.sum():,}", f"{int(elig.sum()):,}")
    return {"X": pd.concat(Xs), "y": np.concatenate(ys), "w": np.concatenate(ws),
            "ts": ts[0].append(ts[1:]) if len(ts) > 1 else ts[0],
            "house": np.concatenate(hs)}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def per_minute_frame(d: dict, pred: np.ndarray, house: int) -> pd.DataFrame:
    """Predictions on the house's full regular index (NaN where not eligible), for daily/cycle metrics."""
    sel = d["house"] == house
    idx = d["ts"][sel]
    full = pd.date_range(idx.min(), idx.max(), freq="1min")
    f = pd.DataFrame(index=full)
    f["true"] = pd.Series(d["y"][sel], index=idx)
    f["pred"] = pd.Series(pred[sel], index=idx)
    f["agg"] = pd.Series(d["X"]["agg"].to_numpy()[sel], index=idx)
    f["eval"] = f["true"].notna()
    return f


def evaluate(d: dict, pred: np.ndarray, params: CycleParams, on_w: float) -> dict[str, Any]:
    out = {"pooled": regression_metrics(d["y"], pred, on_w)}
    for h in np.unique(d["house"]):
        sel = d["house"] == h
        r = regression_metrics(d["y"][sel], pred[sel], on_w)
        f = per_minute_frame(d, pred, h)
        r["daily_energy_mae_kwh"] = daily_energy_mae(f.fillna({"true": 0, "pred": 0}), "true", "pred")
        r |= cycle_metrics(f["true"], f["pred"], params)
        out[f"house_{h}"] = r
    hk = [k for k in out if k.startswith("house_")]
    out["pooled"]["daily_energy_mae_kwh"] = float(np.mean([out[k]["daily_energy_mae_kwh"] for k in hk]))
    out["pooled"]["cycle_recall"] = float(np.nanmean([out[k]["cycle_recall"] for k in hk]))
    out["pooled"]["cycle_precision"] = float(np.nanmean([out[k]["cycle_precision"] for k in hk]))
    return out


def tune_threshold(model: LGBMHurdle, d: dict, on_w: float) -> tuple[float, pd.DataFrame]:
    prob, power = model.predict_parts(d["X"])
    rows = []
    for thr in np.round(np.arange(0.1, 0.91, 0.05), 2):
        p = np.where(prob >= thr, power, 0.0)
        r = regression_metrics(d["y"], p, on_w)
        rows.append({"threshold": thr, **r})
    t = pd.DataFrame(rows)
    return float(t.loc[t["f1"].idxmax(), "threshold"]), t


def flatten(results: dict[str, dict], split: str) -> pd.DataFrame:
    rows = []
    for model, res in results.items():
        for scope, r in res.items():
            rows.append({"split": split, "model": model, "scope": scope, **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pick_window(f: pd.DataFrame, days: int = 3) -> pd.Timestamp:
    """Start of the ``days``-day local window with complete data and the most true active minutes (<= 2 cycles/day)."""
    loc = f.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    day = pd.DataFrame({"cov": f["eval"].groupby(loc).mean(), "on": (f["true"] >= 20).groupby(loc).sum()})
    roll_cov = day["cov"].rolling(days).min()
    roll_on = day["on"].rolling(days).sum()
    score = roll_on.where((roll_cov > 0.98) & (day["on"].rolling(days).min() > 0))
    end = score.idxmax() if score.notna().any() else day.index[days]
    return end - pd.Timedelta(days=days - 1)


def plot_actual_vs_pred(f: pd.DataFrame, house: int, model_name: str, baseline: pd.Series, path) -> None:
    start = _pick_window(f).tz_localize(LOCAL_TZ)
    w = f.loc[start.tz_convert("UTC"): (start + pd.Timedelta(days=3)).tz_convert("UTC")].copy()
    b = baseline.reindex(w.index)
    w.index = w.index.tz_convert(LOCAL_TZ).tz_localize(None)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    axes[0].plot(w.index, w["agg"], color=MUTED, lw=0.7)
    axes[0].set_ylabel("Aggregate (W)\nmodel input")
    axes[0].set_title(f"Unseen test House {house}: actual vs predicted washing-machine power")
    axes[1].plot(w.index, w["true"], color=INK_2, lw=1.3, label="Actual (sub-metered)")
    axes[1].plot(w.index, w["pred"], color=BLUE, lw=1.2, label=f"Predicted: {model_name}")
    axes[1].plot(w.index, b.to_numpy(), color=ORANGE, lw=1.0, label="Usage-profile baseline")
    axes[1].set_ylabel("Washing machine (W)")
    axes[1].legend(loc="upper right")
    time_axis(axes[1], "%a %H:%M")
    axes[1].set_xlabel("Local time")
    save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path) -> None:
    t = table[(table["split"] == "test") & (table["scope"] == "pooled")].set_index("model")
    t["abs_energy_error_pct"] = t["energy_error_pct"].abs()
    specs = [("mae_w", "MAE, all minutes (W)  lower=better"), ("mae_on_w", "MAE when WM on (W)  lower=better"),
             ("f1", "On/off F1  higher=better"), ("daily_energy_mae_kwh", "Daily energy MAE (kWh)  lower=better")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (col, label) in zip(axes, specs):
        vals = t[col]
        ax.barh(t.index, vals, color=BLUE, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3g}", va="center", fontsize=8, color=INK_2)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, vals.max() * 1.3 if vals.max() > 0 else 1)
    axes[0].invert_yaxis()
    fig.suptitle("Test houses (unseen): model comparison", fontweight="bold")
    save(fig, path)


def plot_daily_scatter(frames: dict[int, pd.DataFrame], path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(5.2 * len(frames), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (h, f) in zip(axes, frames.items()):
        d = daily_energy(f.fillna({"true": 0, "pred": 0}), ["true", "pred"])
        mx = max(d["true"].max(), d["pred"].max()) * 1.05
        ax.plot([0, mx], [0, mx], color=MUTED, ls="--", lw=1)
        ax.scatter(d["true"], d["pred"], s=12, color=BLUE, alpha=0.6)
        r = d["true"].corr(d["pred"])
        ax.set_title(f"House {h}: {len(d)} days, r = {r:.2f}")
        ax.set_xlabel("Actual WM energy per day (kWh)")
        ax.set_ylabel("Predicted (kWh)")
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
    fig.suptitle("Daily washing-machine energy on unseen test houses (dashed = perfect)", fontweight="bold")
    save(fig, path)


def plot_residuals(frames: dict[int, pd.DataFrame], path) -> None:
    f = pd.concat(frames.values()).dropna(subset=["true", "pred"])
    bins = [-1, 5, 20, 100, 500, 1500, 5000]
    labels = ["0-5", "5-20", "20-100", "100-500", "500-1500", ">1500"]
    f["bin"] = pd.cut(f["true"], bins, labels=labels)
    res = f["pred"] - f["true"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    data = [res[f["bin"] == b].to_numpy() for b in labels]
    bp = axes[0].boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
    for b in bp["boxes"]:
        b.set(facecolor="#d6e6f8", edgecolor=BLUE)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_xlabel("Actual WM power bin (W)")
    axes[0].set_ylabel("Residual: predicted - actual (W)")
    axes[0].set_title("Residuals by true power (whiskers 1.5 IQR)")
    for i, b in enumerate(labels):
        axes[0].text(i + 1, axes[0].get_ylim()[0], f"n={len(data[i]):,}", ha="center", va="bottom", fontsize=7, color=INK_2)
    hour = f.index.tz_convert(LOCAL_TZ).hour
    by_h = pd.DataFrame({"true": f["true"], "pred": f["pred"], "h": hour}).groupby("h")[["true", "pred"]].mean()
    axes[1].plot(by_h.index, by_h["true"], color=INK_2, marker="o", ms=4, label="Actual mean")
    axes[1].plot(by_h.index, by_h["pred"], color=BLUE, marker="o", ms=4, label="Predicted mean")
    axes[1].set_xlabel("Local hour")
    axes[1].set_ylabel("Mean WM power (W)")
    axes[1].set_title("Mean actual vs predicted by hour of day (test houses)")
    axes[1].legend()
    save(fig, path)


def plot_failures(frames: dict[int, pd.DataFrame], path) -> list[dict]:
    """Worst over-estimate day and worst under-estimate day across test houses."""
    days = []
    for h, f in frames.items():
        d = daily_energy(f.fillna({"true": 0, "pred": 0}), ["true", "pred"])
        d["err"] = d["pred"] - d["true"]
        d["house"] = h
        days.append(d)
    d = pd.concat(days)
    picks = [("Over-estimate (phantom load)", d["err"].idxmax(), d.loc[d["err"].idxmax()]),
             ("Under-estimate (missed wash)", d["err"].idxmin(), d.loc[d["err"].idxmin()])]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5))
    out = []
    for ax, (label, day, row) in zip(axes, picks):
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        h = int(row["house"])
        f = frames[h]
        start = pd.Timestamp(day).tz_localize(LOCAL_TZ)
        w = f.loc[start.tz_convert("UTC"): (start + pd.Timedelta(days=1)).tz_convert("UTC")].copy()
        w.index = w.index.tz_convert(LOCAL_TZ).tz_localize(None)
        ax.fill_between(w.index, w["agg"], color=MUTED, alpha=0.3, lw=0, label="Aggregate")
        ax.plot(w.index, w["true"], color=INK_2, lw=1.3, label="Actual WM")
        ax.plot(w.index, w["pred"], color=RED if "Over" in label else VIOLET, lw=1.2, label="Predicted WM")
        ax.set_title(f"{label}: House {h}, {pd.Timestamp(day):%a %d %b %Y}  "
                     f"(actual {row['true']:.2f} kWh, predicted {row['pred']:.2f} kWh)", fontsize=10)
        ax.set_ylabel("Power (W)")
        ax.set_ylim(0, None)
        time_axis(ax)
        ax.legend(loc="upper right", ncol=3)
        out.append({"case": label, "house": h, "day": str(pd.Timestamp(day).date()),
                    "true_kwh": round(float(row["true"]), 3), "pred_kwh": round(float(row["pred"]), 3)})
    fig.subplots_adjust(hspace=0.4)
    save(fig, path)
    return out


def plot_importance(imp: pd.Series, path, top: int = 15) -> None:
    s = (imp / imp.sum() * 100).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.to_numpy(), color=BLUE, height=0.6)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title(f"Top {top} features (on/off classifier of the selected model)")
    save(fig, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_and_select(cfg, causal: bool, rng) -> dict[str, Any]:
    paths, mc = cfg["_paths"], cfg["model"]
    params = CycleParams.from_config(cfg["cycle_detection"])
    on_w = mc["active_threshold_w"]
    sample = {"on_w": on_w, "context": mc["context_minutes"], "frac": mc["off_sample_fraction"]}
    tr = assemble(paths, mc["train_houses"], causal, sample, rng)
    va = assemble(paths, mc["val_houses"], causal)

    models: dict[str, Any] = {
        "Always-off": AlwaysOff(), "Constant mean": ConstantMean(), "Usage profile": UsageProfile(),
        "LightGBM regression": LGBMRegression(mc["random_state"]),
        "LightGBM hurdle": LGBMHurdle(on_w, mc["random_state"]),
    }
    for name, model in models.items():
        log.info("fitting %s (causal=%s)", name, causal)
        model.fit(tr["X"], tr["y"], tr["w"], tr["ts"], X_val=va["X"], y_val=va["y"])
    thr, thr_table = tune_threshold(models["LightGBM hurdle"], va, on_w)
    models["LightGBM hurdle"].threshold = thr
    log.info("hurdle threshold tuned on validation: %.2f", thr)

    val_res = {n: evaluate(va, m.predict(va["X"], va["ts"]), params, on_w) for n, m in models.items()}
    candidates = ["LightGBM regression", "LightGBM hurdle"]
    best = min(candidates, key=lambda n: val_res[n]["pooled"]["daily_energy_mae_kwh"])
    del tr
    return {"models": models, "val": va, "val_res": val_res, "best": best, "threshold": thr, "thr_table": thr_table}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    paths, mc = cfg["_paths"], cfg["model"]
    params = CycleParams.from_config(cfg["cycle_detection"])
    on_w = mc["active_threshold_w"]
    rng = np.random.default_rng(mc["random_state"])

    # ---- main experiment: centred (offline) features -------------------------------------
    sel = train_and_select(cfg, causal=False, rng=rng)
    models, best = sel["models"], sel["best"]
    sel["thr_table"].to_csv(paths.tables / "task3_hurdle_threshold_validation.csv", index=False)
    log.info("selected on validation (daily energy MAE): %s", best)
    del sel["val"]

    te = assemble(paths, mc["test_houses"], causal=False)
    preds = {n: m.predict(te["X"], te["ts"]) for n, m in models.items()}
    test_res = {n: evaluate(te, p, params, on_w) for n, p in preds.items()}
    table = pd.concat([flatten(sel["val_res"], "validation"), flatten(test_res, "test")], ignore_index=True)

    # ---- ablation: causal features only (same pipeline, same selection rule) -------------
    if mc.get("causal_ablation", True):
        rng_c = np.random.default_rng(mc["random_state"])
        sel_c = train_and_select(cfg, causal=True, rng=rng_c)
        del sel_c["val"]
        te_c = assemble(paths, mc["test_houses"], causal=True)
        m_c = sel_c["models"][best]
        res_c = {f"{best} [causal features]": evaluate(te_c, m_c.predict(te_c["X"], te_c["ts"]), params, on_w)}
        val_c = {f"{best} [causal features]": sel_c["val_res"][best]}
        table = pd.concat([table, flatten(val_c, "validation"), flatten(res_c, "test")], ignore_index=True)
        del te_c, sel_c

    # ---- seen-house temporal holdout (how much does knowing the house help?) -------------
    if mc.get("temporal_experiment", True):
        split = mc["temporal_split_fraction"]
        rng_t = np.random.default_rng(mc["random_state"])
        sample = {"on_w": on_w, "context": mc["context_minutes"], "frac": mc["off_sample_fraction"]}
        tr_t = assemble(paths, mc["train_houses"], False, sample, rng_t, time_slice=(0, split))
        te_t = assemble(paths, mc["train_houses"], False, None, None, time_slice=(split, 1))
        # No validation house here, so reuse the tree count early stopping found in the main run.
        main = models[best]
        if best == "LightGBM regression":
            m_t = LGBMRegression(mc["random_state"], n_estimators=main.model_.best_iteration_ or 500)
        else:
            n = max(main.clf_.best_iteration_ or 500, main.reg_.best_iteration_ or 500)
            m_t = LGBMHurdle(on_w, mc["random_state"], threshold=main.threshold, n_estimators=n)
        m_t.fit(tr_t["X"], tr_t["y"], tr_t["w"], tr_t["ts"])
        res_t = {f"{best} [seen houses, later period]": evaluate(te_t, m_t.predict(te_t["X"]), params, on_w),
                 "Always-off [seen houses, later period]": evaluate(te_t, np.zeros(len(te_t["y"])), params, on_w)}
        table = pd.concat([table, flatten(res_t, "temporal_holdout")], ignore_index=True)
        del tr_t, te_t

    table = table.round(4)
    table.to_csv(paths.tables / "task3_metrics.csv", index=False)

    # ---- figures and artefacts on the test houses --------------------------------------
    frames = {h: per_minute_frame(te, preds[best], h) for h in mc["test_houses"]}
    base_frames = {h: per_minute_frame(te, preds["Usage profile"], h) for h in mc["test_houses"]}
    for h, f in frames.items():
        f.assign(pred_usage_profile=base_frames[h]["pred"]).to_parquet(paths.data / f"test_predictions_house{h}.parquet")
    h0 = mc["test_houses"][0]
    plot_actual_vs_pred(frames[h0], h0, best, base_frames[h0]["pred"], paths.figures / "task3_actual_vs_predicted.png")
    plot_model_comparison(table, paths.figures / "task3_model_comparison.png")
    plot_daily_scatter(frames, paths.figures / "task3_daily_energy_scatter.png")
    plot_residuals(frames, paths.figures / "task3_residuals.png")
    failures = plot_failures(frames, paths.figures / "task3_failure_cases.png")
    imp = models[best].feature_importance()
    imp.to_csv(paths.tables / "task3_feature_importance.csv", header=["gain"])
    plot_importance(imp, paths.figures / "task3_feature_importance.png")

    summary = {"selected_model": best, "hurdle_threshold": sel["threshold"], "failure_days": failures,
               "test_pooled": test_res[best]["pooled"], "n_features": int(te["X"].shape[1])}
    with open(paths.tables / "task3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    view = table[table["scope"] == "pooled"][["split", "model", "mae_w", "rmse_w", "mae_on_w", "energy_error_pct",
                                               "precision", "recall", "f1", "daily_energy_mae_kwh", "cycle_recall",
                                               "cycle_precision"]]
    log.info("Pooled metrics:\n%s", view.to_string())
    return summary
