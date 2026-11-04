"""Baselines and LightGBM models for washing-machine power estimation.

All models share ``fit(X, y, w, meta)`` / ``predict(X, meta)`` where ``meta`` carries the
timestamps (needed by the usage-profile baseline). Predictions are in W, clipped at 0.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
LOCAL_TZ = "Europe/London"


class AlwaysOff:
    name = "Always-off"

    def fit(self, X, y, w=None, meta=None, X_val=None, y_val=None):
        return self

    def predict(self, X, meta=None):
        return np.zeros(len(X))


class ConstantMean:
    """Predicts the (weighted) training-mean WM power every minute."""
    name = "Constant mean"

    def fit(self, X, y, w=None, meta=None, X_val=None, y_val=None):
        self.value_ = float(np.average(y, weights=w))
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), self.value_)


class UsageProfile:
    """Mean WM power by local weekday x hour across the training houses (a usage-profile baseline)."""
    name = "Usage profile (weekday x hour)"

    @staticmethod
    def _key(ts: pd.DatetimeIndex) -> np.ndarray:
        local = ts.tz_convert(LOCAL_TZ)
        return np.asarray(local.day_of_week * 24 + local.hour)

    def fit(self, X, y, w=None, meta=None, X_val=None, y_val=None):
        k = self._key(meta)
        w = np.ones(len(y)) if w is None else w
        num = np.bincount(k, weights=y * w, minlength=168)
        den = np.bincount(k, weights=w, minlength=168)
        self.table_ = np.where(den > 0, num / np.maximum(den, 1e-9), np.average(y, weights=w))
        return self

    def predict(self, X, meta=None):
        return self.table_[self._key(meta)]


LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=-1)


class LGBMRegression:
    """Single gradient-boosted regressor of WM power (squared-error objective)."""
    name = "LightGBM regression"

    def __init__(self, random_state: int = 42, **params):
        self.params = LGB_PARAMS | params | {"random_state": random_state}

    def fit(self, X, y, w=None, meta=None, X_val=None, y_val=None):
        self.model_ = lgb.LGBMRegressor(**self.params)
        kw = {}
        if X_val is not None:
            kw = dict(eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        self.model_.fit(X, y, sample_weight=w, **kw)
        log.info("%s: %d trees", self.name, self.model_.best_iteration_ or self.params["n_estimators"])
        return self

    def predict(self, X, meta=None):
        return np.clip(self.model_.predict(X), 0, None)

    def feature_importance(self) -> pd.Series:
        b = self.model_.booster_
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)


class LGBMHurdle:
    """Two-part model: P(active) classifier x E[power | active] regressor.

    ``threshold`` (tuned on validation) turns the probability into an on/off gate:
    prediction = E[power | active] if P >= threshold else 0. With threshold=None the
    expectation P x E is returned (MSE-optimal but smears low power over off minutes).
    """
    name = "LightGBM hurdle"

    def __init__(self, on_w: float = 20, random_state: int = 42, threshold: float | None = 0.5, **params):
        self.on_w, self.threshold = on_w, threshold
        self.params = LGB_PARAMS | params | {"random_state": random_state}

    def fit(self, X, y, w=None, meta=None, X_val=None, y_val=None):
        on = y >= self.on_w
        self.clf_ = lgb.LGBMClassifier(**self.params)
        self.reg_ = lgb.LGBMRegressor(**self.params)
        kw_c, kw_r = {}, {}
        if X_val is not None:
            on_val = y_val >= self.on_w
            kw_c = dict(eval_set=[(X_val, on_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
            kw_r = dict(eval_set=[(X_val[on_val], y_val[on_val])], callbacks=[lgb.early_stopping(100, verbose=False)])
        self.clf_.fit(X, on, sample_weight=w, **kw_c)
        self.reg_.fit(X[on], y[on], sample_weight=None if w is None else w[on], **kw_r)
        log.info("%s: classifier %s trees, regressor %s trees", self.name,
                 self.clf_.best_iteration_ or self.params["n_estimators"], self.reg_.best_iteration_ or self.params["n_estimators"])
        return self

    def predict_parts(self, X) -> tuple[np.ndarray, np.ndarray]:
        return self.clf_.predict_proba(X)[:, 1], np.clip(self.reg_.predict(X), 0, None)

    def predict(self, X, meta=None, threshold: float | None = "default"):
        thr = self.threshold if threshold == "default" else threshold
        prob, power = self.predict_parts(X)
        return prob * power if thr is None else np.where(prob >= thr, power, 0.0)

    def feature_importance(self) -> pd.Series:
        b = self.clf_.booster_
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)
