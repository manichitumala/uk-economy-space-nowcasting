"""Model zoo: naive baselines, a conventional bridge equation and regularised / boosted learners.

All estimators follow the scikit-learn fit/predict API so the walk-forward engine can treat
them uniformly. Tree models are deliberately shallow: the training sets are small
(a few hundred region-quarters) and strongly cross-correlated.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ConstantModel(BaseEstimator, RegressorMixin):
    """Predict a constant: 0 or the training mean."""

    def __init__(self, kind: str = "mean"):
        self.kind = kind

    def fit(self, X, y):
        self.value_ = 0.0 if self.kind == "zero" else float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class ColumnModel(BaseEstimator, RegressorMixin):
    """Predict the value of one feature column (e.g. last published growth)."""

    def __init__(self, column: int = 0):
        self.column = column

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.fill_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        v = np.asarray(X, float)[:, self.column]
        return np.where(np.isfinite(v), v, self.fill_)


def _linear(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def ridge():
    return _linear(RidgeCV(alphas=np.logspace(-2, 4, 25)))


def elastic_net():
    return _linear(ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0], alphas=40,
                                cv=TimeSeriesSplit(4), max_iter=20000))


def bridge():
    """Conventional OLS bridge equation (used with a handful of macro indicators)."""
    return _linear(LinearRegression())


def xgboost(seed: int = 0):
    return XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=10, reg_lambda=10.0,
                        objective="reg:pseudohubererror", random_state=seed, n_jobs=4,
                        verbosity=0)


def lightgbm(seed: int = 0):
    return LGBMRegressor(n_estimators=300, num_leaves=4, learning_rate=0.03,
                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=10.0, objective="huber",
                         random_state=seed, verbose=-1, n_jobs=4)
