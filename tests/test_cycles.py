import numpy as np
import pandas as pd
import pytest

from wattwise.cycles import CycleParams, cycle_mask, detect_cycles

P = CycleParams(on_threshold_w=20, max_gap_minutes=10, min_duration_minutes=15, min_energy_wh=50,
                min_peak_w=300, max_duration_minutes=360)


def _series(values, start="2014-01-01"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range(start, periods=len(values), freq="1min"))


def _wash(heat=15, tumble=40, pause=5, spin=10):
    return [2000] * heat + [100] * tumble + [0] * pause + [100] * 10 + [400] * spin


def test_single_cycle_statistics():
    vals = [0] * 30 + _wash() + [0] * 30
    c = detect_cycles(_series(vals), P)
    assert len(c) == 1 and c["accepted"].iloc[0]
    row = c.iloc[0]
    assert row["duration_min"] == 80 and row["active_min"] == 75 and row["peak_w"] == 2000
    expected_wh = (15 * 2000 + 50 * 100 + 10 * 400) / 60
    assert row["energy_wh"] == pytest.approx(expected_wh)
    assert row["mean_active_w"] == pytest.approx(expected_wh * 60 / 75)
    assert row["start"] == pd.Timestamp("2014-01-01 00:30")


def test_pauses_merged_but_long_gaps_split():
    vals = [0] * 5 + _wash() + [0] * 11 + _wash() + [0] * 5
    c = detect_cycles(_series(vals), P)
    assert c["accepted"].sum() == 2
    vals = [0] * 5 + _wash() + [0] * 10 + _wash() + [0] * 5
    assert detect_cycles(_series(vals), P)["accepted"].sum() == 1


def test_isolated_spikes_and_short_events_rejected():
    vals = [0] * 20 + [2500] + [0] * 30 + [150] * 8 + [0] * 30 + [90] * 40 + [0] * 5  # 60 Wh but never above 300 W
    c = detect_cycles(_series(vals), P)
    assert not c["accepted"].any()
    assert c["reject_reason"].tolist() == ["too_short", "too_short", "low_peak"]


def test_too_long_event_rejected_and_nan_handling():
    vals = [np.nan] * 3 + [500] * 400 + [np.nan] * 3
    c = detect_cycles(_series(vals), P)
    assert c["reject_reason"].tolist() == ["too_long"]
    vals = [0] * 5 + [2000] * 20 + [np.nan] * 4 + [100] * 20 + [0] * 5
    c = detect_cycles(_series(vals), P)
    assert c["accepted"].iloc[0] and c["missing_min"].iloc[0] == 4 and c["duration_min"].iloc[0] == 44


def test_empty_and_all_off_inputs():
    assert detect_cycles(pd.Series(dtype=float), P).empty
    assert detect_cycles(_series([0] * 100), P).empty


def test_irregular_index_rejected():
    s = pd.Series([0, 100, 0], index=pd.to_datetime(["2014-01-01 00:00", "2014-01-01 00:01", "2014-01-01 00:05"]))
    with pytest.raises(ValueError):
        detect_cycles(s, P)


def test_cycle_mask_and_config():
    s = _series([0] * 10 + _wash() + [0] * 10)
    m = cycle_mask(s.index, detect_cycles(s, P))
    assert m.sum() == 80 and not m.iloc[:10].any()
    p = CycleParams.from_config({"on_threshold_w": 15, "unused": 1})
    assert p.on_threshold_w == 15 and p.max_gap_minutes == 10
