"""Task 2 stage: washing-machine availability, cycles, usage patterns and figures."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .clean_data import flag_stuck
from .cycles import CycleParams, detect_cycles
from .eda import (LOCAL_TZ, availability_summary, cycle_characteristics, enrich_cycles, prepare_wm,
                  usage_tables)
from .io import clean_house_path, read_clean_house
from .plotting import BLUE, INK_2, MUTED, ORANGE, RED, save, time_axis
from .stage_prepare import load_cached

log = logging.getLogger(__name__)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def screen_all_channels(cfg: dict[str, Any], params: CycleParams) -> pd.DataFrame:
    """Quick quality screen of every house's washing-machine channel (official file, no time-base repair)."""
    paths, wcfg = cfg["_paths"], cfg["washing_machine"]
    rows = []
    for house, col in wcfg["all_channels"].items():
        d = read_clean_house(clean_house_path(paths.clean_dir, int(house)), ["Unix", col])
        s = d.set_index(pd.to_datetime(d["Unix"], unit="s"))[col].astype("float32")
        m = s.resample("1min").mean()
        stuck = flag_stuck(m, wcfg["stuck_minutes"])
        wm = m.where(~stuck)
        cyc = detect_cycles(wm, params)
        acc = cyc[cyc["accepted"]]
        obs_days = m.notna().sum() / 1440
        rows.append({"house": int(house), "wm_column": col, "first": m.index.min().date(), "last": m.index.max().date(),
                     "coverage_pct": round(100 * m.notna().mean(), 1), "active_ge20w_pct": round(100 * (wm >= 20).mean(), 2),
                     "stuck_minutes": int(stuck.sum()), "accepted_cycles": len(acc),
                     "cycles_per_week": round(len(acc) / obs_days * 7, 2),
                     "median_cycle_wh": round(float(acc["energy_wh"].median()), 0) if len(acc) else np.nan,
                     "selected": int(house) in wcfg["channels"]})
        log.info("screen house %s: %s", house, rows[-1])
    return pd.DataFrame(rows)


def sensitivity(houses_wm: dict[int, pd.Series], params: CycleParams, thresholds, gaps) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        for gap in gaps:
            p = replace(params, on_threshold_w=thr, max_gap_minutes=gap)
            for h, wm in houses_wm.items():
                c = detect_cycles(wm, p)
                rows.append({"on_threshold_w": thr, "max_gap_minutes": gap, "house": h,
                             "cycles": int(c["accepted"].sum())})
    t = pd.DataFrame(rows).pivot_table(index=["on_threshold_w", "max_gap_minutes"], columns="house", values="cycles")
    t["total"] = t.sum(axis=1)
    return t


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pick_trace_day(m: pd.DataFrame, cycles: pd.DataFrame) -> pd.Timestamp:
    """A local day with full coverage, exactly one accepted cycle, energy closest to the house median."""
    loc_day = cycles["start_local"].dt.normalize()
    one = cycles[loc_day.map(loc_day.value_counts()) == 1].copy()
    obs = (m["n_rows"] > 0).groupby(m.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()).mean()
    one["day"] = one["start_local"].dt.normalize()
    one = one[one["day"].map(obs).fillna(0) >= 0.99]
    if one.empty:
        one = cycles.assign(day=cycles["start_local"].dt.normalize())
    med = cycles["energy_wh"].median()
    return one.loc[(one["energy_wh"] - med).abs().idxmin(), "day"]


def plot_trace(m: pd.DataFrame, cycles: pd.DataFrame, house: int, path) -> str:
    day = _pick_trace_day(m, cycles)
    start = pd.Timestamp(day).tz_localize(LOCAL_TZ)
    w = m.loc[start.tz_convert("UTC"): (start + pd.Timedelta(days=1)).tz_convert("UTC")].copy()
    w.index = w.index.tz_convert(LOCAL_TZ).tz_localize(None)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(w.index, w["aggregate"], color=MUTED, alpha=0.35, lw=0, label="Household aggregate")
    ax.plot(w.index, w["aggregate"], color=MUTED, lw=0.8)
    ax.plot(w.index, w["wm"], color=BLUE, lw=1.5, label="Washing machine (sub-metered)")
    c = cycles[cycles["start_local"].dt.normalize() == day].iloc[0]
    ax.axvspan(c["start_local"], c["start_local"] + pd.Timedelta(minutes=int(c["duration_min"])), color=BLUE, alpha=0.08,
               label=f"Detected cycle: {c['duration_min']:.0f} min, {c['energy_wh'] / 1000:.2f} kWh")
    ax.set_ylabel("Power (W, 1-min mean)")
    ax.set_xlabel("Local time (Europe/London)")
    ax.set_title(f"House {house}: 24-hour aggregate and washing-machine power, {pd.Timestamp(day):%a %d %b %Y}")
    ax.set_xlim(w.index.min(), w.index.min() + pd.Timedelta(days=1))
    ax.set_ylim(0, None)
    time_axis(ax)
    ax.legend(loc="upper left")
    save(fig, path)
    return str(pd.Timestamp(day).date())


def plot_example_cycles(data: dict[int, tuple[pd.Series, pd.DataFrame]], houses: list[int], params: CycleParams, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    for ax, h in zip(axes.ravel(), houses):
        wm, cyc = data[h]
        ok = cyc[cyc["missing_min"] == 0]
        c = ok.loc[(ok["energy_wh"] - cyc["energy_wh"].median()).abs().idxmin()]
        s = wm.loc[c["start"] - pd.Timedelta(minutes=15): c["end"] + pd.Timedelta(minutes=15)]
        t = (s.index - c["start"]).total_seconds() / 60
        ax.plot(t, s.to_numpy(), color=BLUE, lw=1.3)
        ax.axvspan(0, c["duration_min"], color=BLUE, alpha=0.08)
        ax.axhline(params.on_threshold_w, color=ORANGE, lw=0.9, ls="--")
        ax.set_title(f"House {h}  |  {c['start_local']:%d %b %Y %H:%M}", fontsize=10)
        ax.text(0.98, 0.95, f"{c['duration_min']:.0f} min\n{c['energy_wh']:.0f} Wh\npeak {c['peak_w']:.0f} W\n"
                            f"heating {100 * c['heating_share']:.0f}% of energy",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
        ax.set_xlabel("Minutes since cycle start")
    for ax in axes[:, 0]:
        ax.set_ylabel("WM power (W)")
    axes[0, 0].text(0.02, 0.9, f"dashed: {params.on_threshold_w:.0f} W activity threshold", transform=axes[0, 0].transAxes,
                    fontsize=8, color=ORANGE)
    fig.suptitle("Detected washing-machine cycles (shaded = detected span; typical cycle per house)", fontweight="bold")
    fig.subplots_adjust(hspace=0.45)
    save(fig, path)


def plot_cross_house(cycles: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    houses = sorted(cycles["house"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specs = [("energy_wh", "Energy per cycle (Wh)"), ("duration_min", "Cycle duration (min)")]
    for ax, (col, label) in zip(axes[:2], specs):
        vals = [cycles.loc[cycles["house"] == h, col].to_numpy() for h in houses]
        bp = ax.boxplot(vals, tick_labels=[str(h) for h in houses], showfliers=False, patch_artist=True, widths=0.6)
        for b in bp["boxes"]:
            b.set(facecolor="#d6e6f8", edgecolor=BLUE)
        for med in bp["medians"]:
            med.set(color=BLUE, lw=2)
        ax.set_title(label)
        ax.set_xlabel("House")
    s = summary.set_index("house").loc[houses]
    axes[2].bar([str(h) for h in houses], s["cycles_per_week"], color=BLUE, width=0.6)
    for i, v in enumerate(s["cycles_per_week"]):
        axes[2].text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    axes[2].set_ylim(0, s["cycles_per_week"].max() * 1.15)
    axes[2].set_title("Cycles per observed week")
    axes[2].set_xlabel("House")
    fig.suptitle("Cross-house comparison of washing-machine cycles (boxes: IQR, whiskers: 1.5 IQR)", fontweight="bold")
    save(fig, path)


def plot_usage(hour: pd.DataFrame, day: pd.DataFrame, cycles: pd.DataFrame, path) -> None:
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.3], height_ratios=[1, 1.25], hspace=0.45, wspace=0.18)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    pooled_h = hour.mean(axis=0)
    ax_h.bar(pooled_h.index, pooled_h.to_numpy(), color=BLUE, width=0.75)
    ax_h.set_xticks(range(0, 24, 2))
    ax_h.set_xlim(-0.6, 23.6)
    ax_h.set_title("Cycle starts by local hour (mean of house shares)")
    ax_h.set_ylabel("% of cycles")
    ax_h.axvspan(15.5, 18.5, color=RED, alpha=0.07)
    ax_h.text(17, pooled_h.max() * 0.98, "16-19h\nnetwork peak", ha="center", va="top", fontsize=8, color=RED)
    pooled_d = day.mean(axis=0)
    ax_d.bar(WEEKDAYS, pooled_d.to_numpy(), color=BLUE, width=0.65)
    ax_d.axhline(100 / 7, color=MUTED, ls="--", lw=0.9)
    ax_d.set_title("By weekday")
    ax_d.text(6.4, 100 / 7 + 0.3, "uniform", fontsize=7, color=INK_2, ha="right")
    ax_hm = fig.add_subplot(gs[1, 0])
    im = ax_hm.imshow(hour.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=30)
    ax_hm.set_yticks(range(len(hour)), [f"H{h}" for h in hour.index])
    ax_hm.set_xticks(range(0, 24, 2), range(0, 24, 2))
    ax_hm.set_xlabel("Local hour of cycle start")
    ax_hm.set_title("Per house (% of that house's cycles; rows sum to 100)")
    ax_hm.grid(False)
    ax_dm = fig.add_subplot(gs[1, 1])
    ax_dm.imshow(day.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=30)
    ax_dm.set_yticks(range(len(day)), [f"H{h}" for h in day.index])
    ax_dm.set_xticks(range(7), [d[0] for d in WEEKDAYS])
    ax_dm.grid(False)
    fig.colorbar(im, ax=[ax_hm, ax_dm], fraction=0.025, pad=0.02, label="% of cycles (capped at 30)", extend="max")
    fig.suptitle(f"When are washes started? {len(cycles):,} detected cycles, {len(hour)} houses", fontweight="bold")
    save(fig, path)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    paths, wcfg, ecfg = cfg["_paths"], cfg["washing_machine"], cfg["eda"]
    params = CycleParams.from_config(cfg["cycle_detection"])
    houses = [int(h) for h in wcfg["channels"]]
    missing = {ecfg["trace_house"], *ecfg["example_cycle_houses"]} - set(houses)
    if missing:
        raise ValueError(f"eda.trace_house / example_cycle_houses {sorted(missing)} are not in washing_machine.channels")

    all_cycles, summaries, data, wms = [], [], {}, {}
    for h in houses:
        m = load_cached(paths, h)
        wm = prepare_wm(m)
        cyc = detect_cycles(wm, params)
        summaries.append(availability_summary(h, m, cyc, on_w=params.on_threshold_w))
        enriched = enrich_cycles(h, cyc, wm, ecfg["heating_w"], tuple(ecfg["peak_hours"]))
        all_cycles.append(enriched)
        data[h] = (wm, enriched)
        wms[h] = wm
        if h == ecfg["trace_house"]:
            trace_m = m
    cycles = pd.concat(all_cycles, ignore_index=True)
    summary = pd.DataFrame(summaries)
    cycles.to_csv(paths.tables / "task2_cycles.csv", index=False)
    summary.to_csv(paths.tables / "task2_availability.csv", index=False)

    hour, day = usage_tables(cycles)
    hour.to_csv(paths.tables / "task2_hour_of_day_pct.csv")
    day.to_csv(paths.tables / "task2_weekday_pct.csv")
    chars = cycle_characteristics(cycles)
    chars.to_csv(paths.tables / "task2_cycle_characteristics.csv")
    sens = sensitivity(wms, params, ecfg["sensitivity_thresholds_w"], ecfg["sensitivity_gaps_min"])
    sens.to_csv(paths.tables / "task2_detection_sensitivity.csv")

    trace_day = plot_trace(trace_m, data[ecfg["trace_house"]][1], ecfg["trace_house"], paths.figures / "task2_24h_trace.png")
    plot_example_cycles(data, ecfg["example_cycle_houses"], params, paths.figures / "task2_example_cycles.png")
    plot_cross_house(cycles, summary, paths.figures / "task2_cross_house.png")
    plot_usage(hour, day, cycles, paths.figures / "task2_usage_patterns.png")

    screen = screen_all_channels(cfg, params)
    screen.to_csv(paths.tables / "task2_wm_channel_screen.csv", index=False)
    log.info("Availability:\n%s", summary.to_string())
    log.info("Characteristics:\n%s", chars.to_string())
    log.info("Sensitivity:\n%s", sens.to_string())
    return {"cycles": len(cycles), "trace_day": trace_day}
