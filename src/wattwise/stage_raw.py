"""Task 1 stage: clean raw House 1, write tables, figures and a comparison with the official clean data."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import CHANNELS, clean_house_path, raw_part_paths, read_clean_house
from .plotting import BLUE, GRID, INK_2, MUTED, ORANGE, RED, save, time_axis
from .raw_cleaning import CleaningResult, clean_raw_house

log = logging.getLogger(__name__)


def _choose_plot_day(res: CleaningResult, wm_col: str) -> pd.Timestamp:
    """Pick the day that best illustrates cleaning: most nulled/filled values, with a washing cycle and >=90% coverage."""
    before, after = res.readings_before_validity, res.readings
    nulled = (before.notna() & after.isna()).sum(axis=1).resample("1D").sum()
    m = res.minutely
    filled = m[[c for c in m.columns if c.endswith("_filled")]].sum(axis=1).resample("1D").sum()
    coverage = m["Aggregate"].notna().resample("1D").mean()
    wm_on = (m[wm_col] > 20).resample("1D").sum()
    score = (nulled.reindex(coverage.index, fill_value=0) + filled).where((coverage >= 0.9) & (wm_on >= 30))
    if score.dropna().empty:
        return coverage.idxmax()
    return score.idxmax()


def compare_with_official(minutely: pd.DataFrame, clean_dir, house: int, cols: list[str]) -> pd.DataFrame:
    """Agreement between our 1-min series and the official cleaned file resampled the same way.

    Lags of +/-60 min are included: if our local->UTC conversion were wrong, agreement in
    summer would peak at a 60-minute lag instead of 0.
    """
    off = read_clean_house(clean_house_path(clean_dir, house), ["Unix", *cols])
    off = off.set_index(pd.to_datetime(off["Unix"], unit="s", utc=True))[cols].astype("float32")
    off_m = off.resample("1min").mean()
    rows = []
    for c in cols:
        ours = minutely[c]
        month = ours.index.strftime("%Y-%m")
        for lag in (-60, 0, 60):
            # lag -60: official(t + 60 min) vs ours(t), i.e. official still on BST wall-clock time.
            j = pd.concat([ours.rename("ours"), off_m[c].shift(lag).rename("official")], axis=1, join="inner")
            j["month"] = month
            j = j.dropna()
            for mo, g in j.groupby("month"):
                rows.append({"channel": c, "month": mo, "lag_min": lag, "minutes": len(g),
                             "pearson_r": round(float(g["ours"].corr(g["official"])), 3),
                             "mae_w": round(float((g["ours"] - g["official"]).abs().mean()), 1)})
    long = pd.DataFrame(rows)
    return long.pivot_table(index=["channel", "month"], columns="lag_min", values=["pearson_r", "mae_w"]).round(3)


def _plot_before_after_day(res: CleaningResult, day: pd.Timestamp, channels: list[str], names: dict, path) -> None:
    start, end = day, day + pd.Timedelta(days=1)
    raw = res.readings_before_validity.loc[start:end]
    cl = res.minutely.loc[start:end]
    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 2.3 * len(channels)), sharex=True)
    for ax, c in zip(axes, channels):
        ax.plot(raw.index, raw[c], color=MUTED, lw=0.6, label="Before: raw readings (~8 s, UTC-aligned)")
        ax.plot(cl.index, cl[c], color=BLUE, lw=1.3, label="After: cleaned 1-min mean")
        bad = raw[c].notna() & res.readings.loc[start:end, c].isna()
        if bad.any():
            ax.scatter(raw.index[bad], raw[c][bad], color=RED, s=14, zorder=3, label="Nulled (impossible value)")
        f = cl[f"{c}_filled"]
        if f.any():
            ax.scatter(cl.index[f], cl[c][f], color=ORANGE, s=10, zorder=3, label="Short gap forward-filled")
        ax.set_ylabel(f"{names.get(c, c)}\n(W)")
        hi = np.nanpercentile(cl[c], 99.9) if cl[c].notna().any() else 1
        ax.set_ylim(0, max(hi * 1.25, 50))
        if bad.any():
            ax.annotate(f"{int(bad.sum())} glitch reading(s) up to {raw[c][bad].max():,.0f} W (off-scale)",
                        xy=(0.01, 0.88), xycoords="axes fraction", fontsize=8, color=RED)
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title(f"House 1 raw vs cleaned, {day:%Y-%m-%d} (UTC)")
    time_axis(axes[-1])
    axes[-1].set_xlabel("Time of day (UTC)")
    save(fig, path)


def _plot_coverage(res: CleaningResult, path) -> None:
    m = res.minutely
    cov = m["Aggregate"].notna().resample("1D").mean() * 100
    rows = m["n_readings"].resample("1D").sum()
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].bar(rows.index, rows / 1000, width=1.0, color=MUTED)
    axes[0].set_ylabel("Raw readings per day\n(thousands)")
    axes[0].set_title("House 1 raw data coverage (after de-duplication)")
    axes[1].bar(cov.index, cov, width=1.0, color=BLUE)
    for _, o in res.outages.iterrows():
        if o["minutes"] >= 360:
            axes[1].axvspan(o["start"], o["end"], color=RED, alpha=0.12, lw=0)
    axes[1].set_ylabel("Minutes with valid\naggregate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].text(0.01, 0.05, "Red bands: outages >= 6 h (left as missing, not filled)", transform=axes[1].transAxes,
                 fontsize=8, color=RED)
    save(fig, path)


def _plot_daily_max(res: CleaningResult, channels: list[str], names: dict, path) -> None:
    b = res.readings_before_validity[channels].resample("1D").max()
    a = res.readings[channels].resample("1D").max()
    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 2.0 * len(channels)), sharex=True)
    for ax, c in zip(axes, channels):
        ax.plot(b.index, b[c], color=RED, lw=0.8, label="Before validity rules")
        ax.plot(a.index, a[c], color=BLUE, lw=1.0, label="After")
        ax.set_yscale("symlog", linthresh=100)
        ax.set_ylabel(f"{names.get(c, c)}\ndaily max (W)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Daily maximum power before and after removing impossible values (symlog scale)")
    save(fig, path)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    paths, rc = cfg["_paths"], cfg["raw_cleaning"]
    house = int(rc["house"])
    names = rc["channel_names"]
    part_files = raw_part_paths(paths.raw_dir, house)
    res = clean_raw_house(part_files, rc, tz=rc.get("timezone", "Europe/London"))

    out = res.minutely
    out.to_parquet(paths.data / f"house{house}_raw_cleaned_1min.parquet")
    res.outages.to_csv(paths.tables / "task1_outages.csv", index=False)
    with open(paths.tables / "task1_cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump(res.log, f, indent=2, default=str)

    ts = pd.DataFrame({k.split("_")[0]: v for k, v in res.log.items() if k.endswith("_timestamps")})
    ts.to_csv(paths.tables / "task1_timestamp_checks.csv")
    val = pd.DataFrame(res.log["validity"]).T
    val.insert(0, "channel", [names.get(c, c) for c in val.index])
    val.to_csv(paths.tables / "task1_validity_by_channel.csv")
    gaps = pd.DataFrame(res.log["gaps"]).T
    gaps.insert(0, "channel", [names.get(c, c) for c in gaps.index])
    gaps.to_csv(paths.tables / "task1_gaps_by_channel.csv")

    plot_ch = rc["plot_channels"]
    wm_col = next(k for k, v in names.items() if v == "Washing Machine")
    day = pd.Timestamp(rc["plot_day"], tz="UTC") if rc.get("plot_day", "auto") != "auto" else _choose_plot_day(res, wm_col)
    _plot_before_after_day(res, day, plot_ch, names, paths.figures / "task1_before_after_day.png")
    _plot_coverage(res, paths.figures / "task1_coverage.png")
    _plot_daily_max(res, plot_ch, names, paths.figures / "task1_daily_max_before_after.png")

    cmp = compare_with_official(out, paths.clean_dir, house, ["Aggregate", wm_col])
    cmp.columns = [f"{a}_lag{b:+d}" for a, b in cmp.columns]
    cmp.to_csv(paths.tables / "task1_vs_official_clean_monthly.csv")
    log.info("Comparison with official cleaned data:\n%s", cmp.to_string())
    summary = {"plot_day": str(day.date()), **res.log["output"], **res.log["outages"]}
    log.info("Task 1 done: %s", summary)
    return summary
