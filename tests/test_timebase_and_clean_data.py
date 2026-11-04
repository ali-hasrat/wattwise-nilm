import numpy as np
import pandas as pd

from wattwise import timebase
from wattwise.clean_data import flag_stuck


def _signal(index, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(200 + rng.gamma(1.0, 300, len(index)), index=index)


def test_bst_days():
    days = timebase.bst_days(pd.DatetimeIndex(["2014-01-10 05:00", "2014-07-10 05:00", "2014-07-10 06:00"]))
    assert days.tolist() == [False, True]


def test_classify_days_detects_utc_and_local_official_files():
    idx = pd.date_range("2014-07-01", "2014-07-05", freq="1min", inclusive="left")
    raw_local = _signal(pd.date_range("2014-06-30", "2014-07-06", freq="1min"))
    # Days 1-2 converted correctly (official t = raw local t + 1h), days 3-4 left on local time.
    utc_part = raw_local.reindex(idx + pd.Timedelta(hours=1)).to_numpy()
    local_part = raw_local.reindex(idx).to_numpy()
    official = pd.Series(np.where(idx < "2014-07-03", utc_part, local_part), index=idx)
    v = timebase.classify_days(official, raw_local)
    assert v["final"].tolist() == ["utc", "utc", "local", "local"]
    s = timebase.summarise(v)
    assert s["bst_days_local_corrected"] == 2 and s["bst_days_already_utc"] == 2


def test_classify_days_gmt_and_undetermined_inheritance():
    idx = pd.date_range("2014-01-10", periods=1440 * 2, freq="1min")
    raw = _signal(idx)
    v = timebase.classify_days(raw, raw)
    assert (v["verdict"] == "gmt").all()
    idx = pd.date_range("2014-07-01", periods=1440 * 3, freq="1min")
    raw = _signal(pd.date_range("2014-06-30", periods=1440 * 5, freq="1min"))
    official = raw.reindex(idx).copy()
    official[(idx >= "2014-07-02") & (idx < "2014-07-03")] = 300.0  # flat -> no correlation possible
    v = timebase.classify_days(official, raw)
    assert v["verdict"].tolist() == ["local", "undetermined", "local"]
    assert v["final"].tolist() == ["local", "local", "local"]


def test_corrected_utc_index_shifts_only_local_days():
    v = pd.DataFrame({"final": ["utc", "local"]}, index=pd.DatetimeIndex(["2014-07-01", "2014-07-02"]))
    ts = pd.Series(pd.to_datetime(["2014-07-01 12:00", "2014-07-02 12:00"]))
    out = timebase.corrected_utc_index(ts, v)
    assert out.tolist() == [pd.Timestamp("2014-07-01 12:00", tz="UTC"), pd.Timestamp("2014-07-02 11:00", tz="UTC")]


def test_flag_stuck_only_flags_long_constant_nonzero_runs():
    s = pd.Series([0.0] * 100 + [150.0] * 70 + [100, 2000, 50] * 30)
    f = flag_stuck(s, min_minutes=60)
    assert f.sum() == 70 and not f.iloc[:100].any()
