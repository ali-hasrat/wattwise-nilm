import numpy as np
import pandas as pd
import pytest

from wattwise.io import CHANNELS, DataValidationError, raw_part_paths, read_raw_part
from wattwise.raw_cleaning import (
    apply_validity_rules,
    clean_raw_house,
    fill_short_gaps,
    find_gaps,
    inspect_timestamps,
    local_to_utc,
    merge_parts,
    resample_minutely,
    sort_and_deduplicate,
)

CFG = {"resample_freq": "1min", "short_gap_max_minutes": 5, "aggregate_max_w": 20000, "appliance_max_w": 4000}


def _frame(ts, agg, app1=None):
    df = pd.DataFrame({c: np.zeros(len(ts), dtype="float32") for c in CHANNELS})
    df["Aggregate"] = np.asarray(agg, dtype="float32")
    if app1 is not None:
        df["Appliance1"] = np.asarray(app1, dtype="float32")
    df.insert(0, "ts", pd.to_datetime(ts, utc=True))
    return df


def _write_raw(path, times, agg):
    unix = (pd.to_datetime(times).as_unit("s").astype("int64")).tolist()
    rows = ["Time,Unix," + ",".join(CHANNELS)]
    for t, u, a in zip(times, unix, agg):
        rows.append(f"{t},{u},{a}," + ",".join(["0"] * 9))
    path.write_text("\n".join(rows) + "\n")


# --- io -----------------------------------------------------------------------
def test_raw_part_paths_handles_both_naming_styles(tmp_path):
    for n in ("RAW_House1_Part2.csv", "RAW_House1_Part1.csv", "RAW_House_13_Part1.csv", "RAW_House11_Part1.csv"):
        (tmp_path / n).write_text("x")
    assert [p.name for p in raw_part_paths(tmp_path, 1)] == ["RAW_House1_Part1.csv", "RAW_House1_Part2.csv"]
    assert [p.name for p in raw_part_paths(tmp_path, 13)] == ["RAW_House_13_Part1.csv"]
    with pytest.raises(FileNotFoundError):
        raw_part_paths(tmp_path, 7)


def test_read_raw_part_rejects_empty_and_bad_schema(tmp_path):
    empty = tmp_path / "e.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError):
        read_raw_part(empty)
    header_only = tmp_path / "h.csv"
    header_only.write_text("Time,Unix," + ",".join(CHANNELS) + "\n")
    with pytest.raises(DataValidationError, match="no data rows"):
        read_raw_part(header_only)
    bad = tmp_path / "b.csv"
    bad.write_text("Time,Unix,Aggregate\n2014-01-01 00:00:00,1388534400,100\n")
    with pytest.raises(DataValidationError, match="missing columns"):
        read_raw_part(bad)


def test_read_raw_part_parses_nan_strings(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("Time,Unix," + ",".join(CHANNELS) + "\n2014-01-01 00:00:00,1388534400,NaN," + ",".join(["5"] * 9) + "\n")
    df = read_raw_part(p)
    assert df["Aggregate"].isna().all() and df["Appliance1"].iloc[0] == 5


# --- timestamps ---------------------------------------------------------------
def test_inspect_timestamps_counts_problems():
    t = ["2014-01-01 00:00:00", "2014-01-01 00:00:08", "2014-01-01 00:00:08", "2014-01-01 00:00:04", "2014-01-01 01:00:00"]
    df = pd.DataFrame({"Time": t, "Unix": pd.to_datetime(t).as_unit("s").astype("int64"), "Aggregate": [1, 2, 2, 3, 4]})
    s = inspect_timestamps(df)
    assert s["backward_jumps"] == 1 and s["repeated_timestamps"] == 1
    assert s["gaps_gt_10min"] == 1 and not s["monotonic"] and s["exact_duplicate_rows"] == 1


def test_local_to_utc_summer_winter_offsets():
    utc, stats = local_to_utc(pd.Series(pd.to_datetime(["2014-01-15 12:00:00", "2014-07-15 12:00:00"])))
    assert utc[0] == pd.Timestamp("2014-01-15 12:00", tz="UTC")
    assert utc[1] == pd.Timestamp("2014-07-15 11:00", tz="UTC")  # BST = UTC+1
    assert stats["nonexistent_local_times"] == 0


def test_local_to_utc_fall_back_resolved_by_file_order():
    # 2013-10-27: clocks go back at 02:00 BST -> 01:00 GMT; logger writes 01:30 twice.
    local = pd.Series(pd.to_datetime(["2013-10-27 01:30:00", "2013-10-27 01:59:59", "2013-10-27 01:00:01", "2013-10-27 01:30:00"]))
    utc, stats = local_to_utc(local)
    assert utc[0] == pd.Timestamp("2013-10-27 00:30", tz="UTC")
    assert utc[2] == pd.Timestamp("2013-10-27 01:00:01", tz="UTC")
    assert utc[3] == pd.Timestamp("2013-10-27 01:30", tz="UTC")
    assert utc.is_monotonic_increasing and stats["ambiguous_resolved_by_order"] == 4


def test_local_to_utc_interleaved_fall_back_is_dropped():
    local = pd.Series(pd.to_datetime(["2014-10-26 00:59:00", "2014-10-26 01:10:00", "2014-10-26 01:10:04", "2014-10-26 02:00:00"]))
    utc, stats = local_to_utc(local)
    assert stats["ambiguous_unresolved_dropped"] == 2 and utc.isna().sum() == 2
    assert utc[3] == pd.Timestamp("2014-10-26 02:00", tz="UTC")


def test_local_to_utc_nonexistent_spring_forward():
    utc, stats = local_to_utc(pd.Series(pd.to_datetime(["2014-03-30 01:30:00"])))
    assert stats["nonexistent_local_times"] == 1 and utc.isna().all()


# --- merge / dedupe ------------------------------------------------------------
def test_merge_parts_later_part_wins_overlap():
    p1 = _frame(["2014-01-01 00:00", "2014-01-01 00:01", "2014-01-01 00:02"], [1, 2, 3])
    p2 = _frame(["2014-01-01 00:01", "2014-01-01 00:03"], [20, 40])
    m, stats = merge_parts([p1, p2])
    assert m["Aggregate"].tolist() == [1, 20, 40] and stats["part1_rows_dropped_overlap"] == 2


def test_sort_and_deduplicate_rules():
    df = _frame(["2014-01-01 00:00:10", "2014-01-01 00:00:00", "2014-01-01 00:00:10", "2014-01-01 00:00:05", "2014-01-01 00:00:05"],
                [10, 1, 10, 5, 6], app1=[1, 1, 1, np.nan, 7])
    out, stats = sort_and_deduplicate(df)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert stats["exact_duplicates_dropped"] == 1 and stats["conflicting_timestamps"] == 1
    row = out.loc[pd.Timestamp("2014-01-01 00:00:05", tz="UTC")]
    assert row["Aggregate"] == 6 and row["Appliance1"] == 7  # last non-null wins


# --- validity -----------------------------------------------------------------
def test_validity_rules_null_impossible_but_keep_real_peaks():
    idx = pd.date_range("2014-01-01", periods=6, freq="8s", tz="UTC")
    df = pd.DataFrame({c: np.zeros(6, dtype="float32") for c in CHANNELS}, index=idx)
    df["Aggregate"] = [300, 0, 10500, 29159, -5, 400]
    df["Appliance5"] = [2300, 65535, 8192, 50, -1, 0]
    out, stats = apply_validity_rules(df, 20000, 4000)
    assert out["Aggregate"].isna().tolist() == [False, True, False, True, True, False]
    assert out["Appliance5"].isna().tolist() == [False, True, True, False, True, False]
    assert out["Aggregate"].iloc[2] == 10500  # electric shower kept
    assert stats["Appliance5"]["above_limit"] == 2 and stats["Aggregate"]["zero_aggregate"] == 1


# --- resample / gaps ----------------------------------------------------------
def test_resample_minutely_mean_and_counts_irregular():
    idx = pd.to_datetime(["2014-01-01 00:00:05", "2014-01-01 00:00:50", "2014-01-01 00:02:10"], utc=True)
    df = pd.DataFrame({"Aggregate": [100.0, 200.0, 50.0]}, index=idx)
    m, n = resample_minutely(df)
    assert len(m) == 3 and m["Aggregate"].iloc[0] == 150 and np.isnan(m["Aggregate"].iloc[1])
    assert n.tolist() == [2, 0, 1]


def test_fill_short_gaps_only_fills_short_runs():
    s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 3.0])
    filled, mask = fill_short_gaps(s, max_len=2)
    assert filled.tolist()[:4] == [1, 1, 1, 2]
    assert filled.iloc[4:8].isna().all()  # long gap untouched, not partially filled
    assert mask.sum() == 2


def test_fill_short_gaps_leading_nan_not_filled():
    filled, mask = fill_short_gaps(pd.Series([np.nan, 1.0, 2.0]), max_len=5)
    assert np.isnan(filled.iloc[0]) and not mask.iloc[0]


def test_find_gaps():
    idx = pd.date_range("2014-01-01", periods=10, freq="1min")
    miss = pd.Series([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], index=idx).astype(bool)
    g = find_gaps(miss, min_len=2)
    assert g["minutes"].tolist() == [2, 4] and g["start"].iloc[1] == idx[4] and g["end"].iloc[1] == idx[7]
    assert find_gaps(pd.Series(False, index=idx)).empty


# --- end-to-end on synthetic files --------------------------------------------
def test_clean_raw_house_end_to_end(tmp_path):
    t1 = pd.date_range("2014-01-01 00:00:00", "2014-01-01 00:20:00", freq="10s")
    t2 = pd.date_range("2014-01-01 00:15:00", "2014-01-01 01:00:00", freq="10s")
    t2 = t2[(t2 < "2014-01-01 00:30:00") | (t2 >= "2014-01-01 00:45:00")]  # 15-minute outage
    t2 = t2[(t2 < "2014-01-01 00:20:00") | (t2 >= "2014-01-01 00:23:00")]  # 3-minute gap
    f1, f2 = tmp_path / "RAW_House1_Part1.csv", tmp_path / "RAW_House1_Part2.csv"
    times1 = [str(t) for t in t1] + [str(t1[3])]  # one exact duplicate row
    _write_raw(f1, times1, [100] * len(t1) + [100])
    _write_raw(f2, [str(t) for t in t2], [200] * len(t2))
    res = clean_raw_house([f1, f2], CFG)
    m = res.minutely
    assert m.index.is_unique and m.index.is_monotonic_increasing
    assert res.log["deduplication"]["exact_duplicates_dropped"] == 1
    assert res.log["merge"]["part1_rows_dropped_overlap"] > 0
    assert m["Aggregate_filled"].sum() == 3  # short gap filled
    assert len(res.outages) == 1 and res.outages["minutes"].iloc[0] == 15
    assert m.loc[pd.Timestamp("2014-01-01 00:16", tz="UTC"), "Aggregate"] == 200  # Part2 wins after overlap


def test_count_isolated_spikes_ignores_sustained_loads():
    from wattwise.raw_cleaning import count_isolated_spikes
    idx = pd.date_range("2014-01-01", periods=12, freq="8s", tz="UTC")
    df = pd.DataFrame({c: np.zeros(12, dtype="float32") for c in CHANNELS}, index=idx)
    df["Appliance1"] = [80, 80, 2048, 80, 80, 0, 0, 0, 0, 0, 0, 0]          # isolated glitch
    df["Appliance5"] = [0, 0, 2300, 2300, 2300, 2300, 0, 0, 0, 0, 0, 0]     # real heater run
    out = count_isolated_spikes(df)
    assert out["Appliance1"] == 1 and out["Appliance5"] == 0


def test_config_loads_and_resolves_relative_paths(tmp_path):
    from wattwise.config import load_config
    cfg = load_config()
    p = cfg["_paths"]
    assert p.raw_dir.name == "REFIT_RAW_081116" and p.raw_dir.parent == p.root
    assert set(cfg["model"]["train_houses"]).isdisjoint(cfg["model"]["test_houses"] + cfg["model"]["val_houses"])
    assert set(cfg["model"]["test_houses"]).isdisjoint(cfg["model"]["val_houses"])
    bad = tmp_path / "c.yaml"
    bad.write_text("paths: {raw_dir: a, clean_dir: b, output_dir: c}\n")
    with pytest.raises(KeyError):
        load_config(bad)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
