import numpy as np
import pandas as pd
import pytest

from wattwise.cycles import CycleParams
from wattwise.features import build_features
from wattwise.metrics import _overlaps, cycle_metrics, daily_energy_mae, regression_metrics
from wattwise.models import AlwaysOff, ConstantMean, LGBMHurdle, LGBMRegression, UsageProfile
from wattwise.stage_model import eligible_mask, subsample_train


def _agg(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-07-01", periods=n, freq="1min", tz="UTC")
    return pd.Series(200 + rng.gamma(1, 100, n), index=idx)


# --- features --------------------------------------------------------------------
def test_features_shape_and_no_target_columns():
    X = build_features(_agg())
    assert len(X) == 600 and X.dtypes.eq("float32").all()
    assert not any("wm" in c for c in X.columns)


def test_causal_features_do_not_use_future():
    a = _agg()
    b = a.copy()
    b.iloc[400:] += 3000  # change the future only
    Xa, Xb = build_features(a, causal=True), build_features(b, causal=True)
    pd.testing.assert_frame_equal(Xa.iloc[:400], Xb.iloc[:400])
    Ca, Cb = build_features(a, causal=False), build_features(b, causal=False)
    assert not Ca.iloc[:400].equals(Cb.iloc[:400])  # centred features do look ahead (by design)


def test_features_handle_missing_and_local_time():
    a = _agg()
    a.iloc[100:130] = np.nan
    X = build_features(a)
    assert X.loc[a.index[110], "missing_31"] > 0
    # 2014-07-01 00:00 UTC = 01:00 BST
    assert X["hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 1 / 24), abs=1e-6)
    with pytest.raises(TypeError):
        build_features(pd.Series([1.0, 2.0]))


def test_heater_step_features():
    a = pd.Series(200.0, index=pd.date_range("2014-01-01", periods=200, freq="1min", tz="UTC"))
    a.iloc[100:115] = 2300
    X = build_features(a)
    assert X.loc[a.index[107], "frac_heater_15"] > 0.9
    assert X.loc[a.index[107], "steps_up_31"] == 1 and X.loc[a.index[107], "steps_down_31"] == 1


# --- metrics ---------------------------------------------------------------------
def test_regression_metrics_always_off_is_flattering_but_has_zero_recall():
    y = np.r_[np.zeros(95), np.full(5, 2000.0)]
    r = regression_metrics(y, np.zeros(100))
    assert r["mae_w"] == pytest.approx(100) and r["recall"] == 0 and r["f1"] == 0
    assert r["energy_error_pct"] == pytest.approx(-100) and r["mae_on_w"] == pytest.approx(2000)
    perfect = regression_metrics(y, y)
    assert perfect["mae_w"] == 0 and perfect["f1"] == 1 and perfect["energy_error_pct"] == 0
    with pytest.raises(ValueError):
        regression_metrics(np.array([]), np.array([]))


def test_daily_energy_mae_and_coverage_filter():
    idx = pd.date_range("2014-01-01", periods=2 * 1440, freq="1min", tz="UTC")
    f = pd.DataFrame({"true": 60.0, "pred": 120.0, "eval": True}, index=idx)  # 60 W all day = 1.44 kWh
    assert daily_energy_mae(f, "true", "pred") == pytest.approx(1.44)
    f.loc[idx[1440:], "eval"] = False  # second day has no coverage -> excluded
    assert daily_energy_mae(f, "true", "pred") == pytest.approx(1.44)


def test_cycle_overlap_metrics():
    a = pd.DataFrame({"start": pd.to_datetime(["2014-01-01 00:00", "2014-01-01 05:00"]),
                      "end": pd.to_datetime(["2014-01-01 01:00", "2014-01-01 06:00"])})
    b = pd.DataFrame({"start": pd.to_datetime(["2014-01-01 00:30"]), "end": pd.to_datetime(["2014-01-01 00:40"])})
    assert _overlaps(a, b).tolist() == [True, False] and _overlaps(b, a).tolist() == [True]
    idx = pd.date_range("2014-01-01", periods=300, freq="1min")
    t = pd.Series(0.0, index=idx)
    t.iloc[50:120] = 2000
    p = t.shift(10).fillna(0)
    m = cycle_metrics(t, p, CycleParams())
    assert m["cycle_recall"] == 1 and m["cycle_precision"] == 1


# --- models ----------------------------------------------------------------------
def _toy(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-06", periods=n, freq="1min", tz="UTC")
    on = (np.arange(n) % 400) < 60
    y = np.where(on, 2000.0, 0.0)
    X = pd.DataFrame({"agg": 200 + y + rng.normal(0, 30, n), "noise": rng.normal(size=n)}, index=idx).astype("float32")
    return X, y, idx


def test_baselines():
    X, y, idx = _toy()
    assert (AlwaysOff().fit(X, y).predict(X) == 0).all()
    assert ConstantMean().fit(X, y).predict(X)[0] == pytest.approx(y.mean())
    up = UsageProfile().fit(X, y, None, idx)
    p = up.predict(X, idx)
    assert p.shape == y.shape and (p >= 0).all() and up.table_.shape == (168,)


def test_lightgbm_models_learn_simple_signal():
    X, y, idx = _toy()
    reg = LGBMRegression(n_estimators=300).fit(X, y)  # needs convergence: residual off-power must fall < 20 W
    assert regression_metrics(y, reg.predict(X))["f1"] > 0.95
    hur = LGBMHurdle(n_estimators=50, threshold=0.5).fit(X, y)
    p = hur.predict(X)
    assert regression_metrics(y, p)["f1"] > 0.95 and (p >= 0).all()
    assert hur.feature_importance().index[0] == "agg"


# --- training data selection -------------------------------------------------------
def test_subsample_train_weights_and_eligibility():
    y = pd.Series(np.r_[np.zeros(500), np.full(10, 500.0), np.zeros(500)])
    elig = pd.Series(True, index=y.index)
    elig.iloc[0:10] = False
    keep, w = subsample_train(y, elig, on_w=20, context=20, frac=0.25, rng=np.random.default_rng(0))
    assert keep[480:530].all()                     # all minutes near activity kept
    assert not keep[0:10].any()                    # ineligible never kept
    assert set(np.unique(w[keep])) <= {1.0, 4.0}
    far = keep & (w == 4.0)
    assert 0.15 < far.sum() / (len(y) - 50) < 0.35  # ~25 % of far minutes sampled


def test_eligible_mask_rules():
    m = pd.DataFrame({"n_rows": [5, 0, 5, 5, 5], "aggregate": [1.0, np.nan, 1, 1, 1], "wm": [0.0, np.nan, 0, 0, np.nan],
                      "wm_stuck": [False, False, True, False, False], "issues": [0, 0, 0, 1, 0]})
    assert eligible_mask(m).tolist() == [True, False, False, False, False]
