"""Task 1: inspection and cleaning of raw REFIT data (designed for House 1).

Pipeline (each step logs how many readings it touched):

1. Timestamp checks       parse ``Time``, compare with ``Unix``, measure ordering and intervals.
2. Local clock -> UTC     raw timestamps are UK wall-clock time; convert with explicit DST handling.
3. Merge parts            Part1 and Part2 overlap; the later part wins from its first timestamp.
4. Sort + de-duplicate    drop exact duplicate rows, then collapse same-timestamp rows
                          to the last non-null value per channel.
5. Validity rules         physically impossible values -> NaN (flagged, never silently zeroed).
6. Resample to 1 minute   mean of valid readings per minute.
7. Gap handling           forward-fill gaps <= ``short_gap_max_minutes``; longer gaps stay NaN
                          and are listed as outages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .io import CHANNELS, read_raw_part

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Timestamp inspection
# ---------------------------------------------------------------------------
def inspect_timestamps(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise timestamp quality for one raw part (no modification)."""
    t = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    unix = df["Unix"]
    diff_tu = (t - pd.to_datetime(unix, unit="s")).dt.total_seconds()
    d = unix.diff().dropna()
    return {
        "rows": int(len(df)),
        "first": str(t.min()),
        "last": str(t.max()),
        "time_unparseable": int(t.isna().sum()),
        "unix_missing": int(unix.isna().sum()),
        "time_vs_unix_abs_diff_le_1s": int((diff_tu.abs() <= 1).sum()),
        "time_vs_unix_abs_diff_gt_1s": int((diff_tu.abs() > 1).sum()),
        "monotonic": bool(unix.is_monotonic_increasing),
        "backward_jumps": int((d < 0).sum()),
        "repeated_timestamps": int((d == 0).sum()),
        "interval_median_s": float(d[d > 0].median()) if (d > 0).any() else float("nan"),
        "interval_p99_s": float(d[d > 0].quantile(0.99)) if (d > 0).any() else float("nan"),
        "gaps_gt_1min": int((d > 60).sum()),
        "gaps_gt_10min": int((d > 600).sum()),
        "gaps_gt_1h": int((d > 3600).sum()),
        "gaps_gt_1day": int((d > 86400).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# 2. Local wall clock -> UTC
# ---------------------------------------------------------------------------
def local_to_utc(local: pd.Series, tz: str = "Europe/London") -> tuple[pd.Series, dict[str, int]]:
    """Convert naive local wall-clock timestamps (in file order) to UTC.

    * Spring-forward times that do not exist locally become NaT.
    * Autumn fall-back times occur twice. If the file shows a backward jump inside that
      hour, rows before the jump are the first (BST) occurrence and rows after it the
      second (GMT) one. If there is no backward jump (both hours were sorted together and
      interleave) the two copies cannot be told apart, so those rows become NaT.
    """
    local = pd.Series(pd.to_datetime(local), copy=False).reset_index(drop=True)
    probe = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    nonexistent = probe.isna() & local.notna() & local.dt.tz_localize(tz, ambiguous=True, nonexistent="NaT").isna()
    ambiguous = probe.isna() & local.notna() & ~nonexistent

    is_dst = np.zeros(len(local), dtype=bool)
    unresolved = np.zeros(len(local), dtype=bool)
    if ambiguous.any():
        amb_idx = np.flatnonzero(ambiguous.to_numpy())
        amb_days = local.iloc[amb_idx].dt.date.to_numpy()
        for day in np.unique(amb_days):
            idx = amb_idx[amb_days == day]
            secs = local.iloc[idx].dt.as_unit("s").astype("int64").to_numpy()
            jumps = np.flatnonzero(np.diff(secs) < -1800)
            if len(jumps) == 1:
                is_dst[idx[: jumps[0] + 1]] = True  # first pass through the hour = BST
            else:
                unresolved[idx] = True

    utc = local.dt.tz_localize(tz, ambiguous=is_dst, nonexistent="NaT").dt.tz_convert("UTC")
    utc[unresolved] = pd.NaT
    stats = {
        "nonexistent_local_times": int(nonexistent.sum()),
        "ambiguous_resolved_by_order": int(ambiguous.sum() - unresolved.sum()),
        "ambiguous_unresolved_dropped": int(unresolved.sum()),
    }
    return utc, stats


# ---------------------------------------------------------------------------
# 3-4. Merge, sort, de-duplicate
# ---------------------------------------------------------------------------
def merge_parts(parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Concatenate parts that each carry a UTC ``ts`` column.

    Rule: a later part takes precedence from its first timestamp onward; rows of an
    earlier part at or after that point are dropped. (REFIT Part2 marks unavailable
    sensors as NaN instead of 0, so it is the more informative source where they overlap.)
    """
    kept, stats = [], {}
    for i, part in enumerate(parts):
        if i + 1 < len(parts):
            cutoff = parts[i + 1]["ts"].min()
            mask = part["ts"] < cutoff
            stats[f"part{i + 1}_rows_dropped_overlap"] = int((~mask).sum())
            part = part[mask]
        kept.append(part)
    return pd.concat(kept, ignore_index=True), stats


def overlap_diagnostics(earlier: pd.DataFrame, later: pd.DataFrame) -> dict[str, Any]:
    """Compare two parts where they overlap: agreement, and physical consistency of each part.

    A reading is physically inconsistent when the plug monitors (valid range only) sum to
    more than the whole-house aggregate (+5 % +20 W tolerance for unsynchronised sensors).
    """
    start, end = later["ts"].min(), earlier["ts"].max()
    if start >= end:
        return {"overlap": False}

    def minutely(p: pd.DataFrame) -> pd.DataFrame:
        w = p[(p["ts"] >= start) & (p["ts"] <= end)].set_index("ts")[CHANNELS].sort_index()
        iam = w[CHANNELS[1:]]
        w[CHANNELS[1:]] = iam.where((iam >= 0) & (iam <= 4000))
        return w.resample("1min").mean()

    a, b = minutely(earlier), minutely(later)
    out: dict[str, Any] = {"overlap": True, "start": str(start), "end": str(end)}
    j = pd.concat([a["Aggregate"], b["Aggregate"]], axis=1, keys=["a", "b"]).dropna()
    daily = j.groupby(j.index.date).apply(lambda g: g["a"].corr(g["b"]))
    out["aggregate_r_1min"] = round(float(j["a"].corr(j["b"])), 3)
    out["days_compared"] = int(daily.notna().sum())
    out["days_daily_r_below_0.9"] = int((daily < 0.9).sum())
    for name, m in (("earlier", a), ("later", b)):
        iam = m[CHANNELS[1:]].sum(axis=1, min_count=1)
        ok = m["Aggregate"].notna() & iam.notna()
        out[f"{name}_iam_sum_exceeds_aggregate_frac"] = round(
            float((iam[ok] > m["Aggregate"][ok] * 1.05 + 20).mean()), 4)
    return out


def sort_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sort by ``ts`` and remove duplicates.

    Rule: (a) exact duplicate rows are dropped; (b) remaining rows that share a timestamp
    but disagree are collapsed into one row holding, per channel, the last non-null value
    in file order (the logger appends the newest reading last).
    """
    stats = {"rows_in": int(len(df))}
    df = df.sort_values("ts", kind="stable")
    exact = df.duplicated(subset=["ts", *CHANNELS])
    stats["exact_duplicates_dropped"] = int(exact.sum())
    df = df[~exact]
    dup_ts = df["ts"].duplicated(keep=False)
    stats["rows_sharing_timestamp"] = int(dup_ts.sum())
    stats["conflicting_timestamps"] = int(df.loc[dup_ts, "ts"].nunique())
    if dup_ts.any():
        merged = df[dup_ts].groupby("ts", sort=True)[CHANNELS].last()
        df = pd.concat([df[~dup_ts].set_index("ts")[CHANNELS], merged]).sort_index()
    else:
        df = df.set_index("ts")[CHANNELS]
    stats["rows_out"] = int(len(df))
    return df, stats


# ---------------------------------------------------------------------------
# 5. Validity rules
# ---------------------------------------------------------------------------
def apply_validity_rules(df: pd.DataFrame, aggregate_max_w: float, appliance_max_w: float,
                         zero_aggregate_is_missing: bool = True) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Null out physically impossible readings and count them per channel.

    * negative power on any channel (sensors measure consumption only);
    * aggregate above ``aggregate_max_w`` (beyond a domestic single-phase supply);
    * aggregate exactly 0 W (a whole house never draws 0 W; Part1 wrote 0 for 'unavailable');
    * appliance above ``appliance_max_w`` (plug monitors are 13 A ~ 3 kW devices; these
      readings are mostly bit-pattern glitches such as 4096, 8192, 32767, 65535).
    Legitimate peaks (e.g. a 10 kW electric shower in the aggregate, a 2.3 kW washing-machine
    heater) are below the limits and are kept.
    """
    out = df.copy()
    stats: dict[str, dict[str, int]] = {}
    for c in CHANNELS:
        x = out[c]
        limit = aggregate_max_w if c == "Aggregate" else appliance_max_w
        neg, high = x < 0, x > limit
        zero = (x == 0) if (c == "Aggregate" and zero_aggregate_is_missing) else pd.Series(False, index=x.index)
        bad = neg | high | zero
        stats[c] = {"negative": int(neg.sum()), "above_limit": int(high.sum()),
                    "zero_aggregate": int(zero.sum()), "missing_before": int(x.isna().sum())}
        out.loc[bad, c] = np.nan
    return out, stats


def count_isolated_spikes(df: pd.DataFrame, min_w: float = 300, ratio: float = 5) -> dict[str, int]:
    """Diagnostic only: single readings > ``min_w`` and > ``ratio`` x both neighbours.

    These mix sensor glitches (512/1024/2048 W bit patterns) with real start-up transients
    (e.g. compressor inrush), so they are reported, not removed.
    """
    out = {}
    for c in CHANNELS:
        x = df[c].dropna()
        base = np.maximum(x.shift(1), x.shift(-1)).clip(lower=10)
        out[c] = int(((x > min_w) & (x > ratio * base)).sum())
    return out


# ---------------------------------------------------------------------------
# 6-7. Resampling and gaps
# ---------------------------------------------------------------------------
def resample_minutely(df: pd.DataFrame, freq: str = "1min") -> tuple[pd.DataFrame, pd.Series]:
    """Mean of valid readings per interval (label = interval start, left-closed).

    Returns the resampled frame (complete regular index) and the number of rows per interval.
    """
    r = df.resample(freq, label="left", closed="left")
    return r.mean().astype("float32"), r.size()


def fill_short_gaps(s: pd.Series, max_len: int) -> tuple[pd.Series, pd.Series]:
    """Forward-fill NaN runs of length <= ``max_len`` that have a preceding value.

    Longer runs are left untouched (not even partially filled). Returns (filled, was_filled).
    """
    na = s.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("size")
    fillable = na & (run_len <= max_len)
    filled = s.where(~fillable, s.ffill())
    was_filled = fillable & filled.notna()
    return filled, was_filled


def find_gaps(missing: pd.Series, min_len: int = 1) -> pd.DataFrame:
    """List runs of True in a boolean series indexed by time: start, end (inclusive), minutes."""
    m = missing.astype(bool)
    if not m.any():
        return pd.DataFrame(columns=["start", "end", "minutes"])
    run_id = (m != m.shift()).cumsum()
    g = m[m].groupby(run_id[m])
    runs = pd.DataFrame({"start": g.apply(lambda x: x.index[0]),
                         "end": g.apply(lambda x: x.index[-1]),
                         "minutes": g.size()}).reset_index(drop=True)
    return runs[runs["minutes"] >= min_len].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class CleaningResult:
    minutely: pd.DataFrame                    # cleaned 1-min data + flag columns
    readings: pd.DataFrame                    # de-duplicated, validated raw readings (UTC index)
    outages: pd.DataFrame                     # aggregate outages longer than the fill limit
    log: dict[str, Any] = field(default_factory=dict)
    readings_before_validity: pd.DataFrame | None = None  # sorted, de-duplicated, UTC; for "before" plots


def clean_raw_house(paths: list, cfg: dict[str, Any], tz: str = "Europe/London") -> CleaningResult:
    """Run the full raw cleaning pipeline over the ordered list of raw part files."""
    report: dict[str, Any] = {"inputs": [str(p) for p in paths]}
    parts = []
    for i, path in enumerate(paths, start=1):
        df = read_raw_part(path)
        report[f"part{i}_timestamps"] = inspect_timestamps(df)
        report[f"part{i}_missing_values"] = {c: int(df[c].isna().sum()) for c in CHANNELS}
        # Unix is the integer source of truth (Time is the same clock rounded differently).
        local = pd.to_datetime(df["Unix"], unit="s")
        df["ts"], tz_stats = local_to_utc(local, tz)
        report[f"part{i}_timezone"] = tz_stats
        df = df[df["ts"].notna()].drop(columns=["Time", "Unix"])
        parts.append(df)
        log.info("Part%d: %s rows after UTC conversion (%s)", i, f"{len(df):,}", tz_stats)

    for i in range(len(parts) - 1):
        report[f"overlap_part{i + 1}_part{i + 2}"] = overlap_diagnostics(parts[i], parts[i + 1])
    merged, merge_stats = merge_parts(parts)
    report["merge"] = merge_stats
    readings, dedup_stats = sort_and_deduplicate(merged)
    report["deduplication"] = dedup_stats
    log.info("Deduplication: %s", dedup_stats)

    before = readings
    readings, validity = apply_validity_rules(readings, cfg["aggregate_max_w"], cfg["appliance_max_w"])
    report["validity"] = validity
    iam_sum = readings[CHANNELS[1:]].sum(axis=1, min_count=1)
    report["iam_sum_exceeds_aggregate_rows"] = int((iam_sum > readings["Aggregate"]).sum())
    report["isolated_spikes_kept"] = count_isolated_spikes(readings)

    minutely, counts = resample_minutely(readings, cfg["resample_freq"])
    out = pd.DataFrame(index=minutely.index)
    out.index.name = "ts_utc"
    out["n_readings"] = counts.astype("int32")
    gap_stats = {}
    max_gap = int(cfg["short_gap_max_minutes"])
    for c in CHANNELS:
        filled, was_filled = fill_short_gaps(minutely[c], max_gap)
        out[c] = filled.astype("float32")
        out[f"{c}_filled"] = was_filled
        gap_stats[c] = {"minutes_missing_after_resample": int(minutely[c].isna().sum()),
                        "minutes_filled_short_gap": int(was_filled.sum()),
                        "minutes_still_missing": int(filled.isna().sum())}
    out["outage"] = out["Aggregate"].isna()
    report["gaps"] = gap_stats
    outages = find_gaps(out["outage"], min_len=max_gap + 1)
    report["outages"] = {"count": int(len(outages)), "total_minutes": int(outages["minutes"].sum()),
                         "longest_minutes": int(outages["minutes"].max()) if len(outages) else 0}
    report["output"] = {"first": str(out.index.min()), "last": str(out.index.max()),
                        "minutes": int(len(out)),
                        "aggregate_coverage": float(out["Aggregate"].notna().mean())}
    log.info("Outages: %s", report["outages"])
    return CleaningResult(minutely=out, readings=readings, outages=outages, log=report,
                          readings_before_validity=before)
