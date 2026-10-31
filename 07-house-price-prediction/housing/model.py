"""Feature engineering, base learners, stacking ensemble and conformal prediction intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler, TargetEncoder

NUMERIC = ["sqft", "beds", "baths", "year_built", "lot_sqft", "condition", "garage_spaces", "has_pool",
           "renovated", "school_rating", "x_km", "y_km", "age", "log_sqft", "log_lot_ratio",
           "dist_downtown", "sqft_per_bed", "angle_downtown"]
TARGET = "price"


class HouseFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age"] = 2026 - X["year_built"]
        X["log_sqft"] = np.log(X["sqft"])
        X["log_lot_ratio"] = np.log(X["lot_sqft"] / X["sqft"])
        X["dist_downtown"] = np.hypot(X["x_km"] - 25, X["y_km"] - 25)
        X["angle_downtown"] = np.arctan2(X["y_km"] - 25, X["x_km"] - 25)
        X["sqft_per_bed"] = X["sqft"] / X["beds"]
        return X


def _prep(linear: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if linear:
        # Splines let a linear model capture non-linear effects like the U-shaped age curve.
        num += [("spline", SplineTransformer(n_knots=6, degree=3)), ("scale", StandardScaler())]
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC),
        ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["property_type"]),
        # Cross-fitted target encoding of the 40 neighbourhoods (no target leakage).
        ("nb", TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=0)), ["neighbourhood"]),
    ])


def _log_target(model) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)


def build_models(seed: int = 0) -> dict:
    base = {
        "ridge_splines": Pipeline([("f", HouseFeatures()), ("p", _prep(True)), ("m", RidgeCV(np.logspace(-2, 3, 20)))]),
        "random_forest": Pipeline([("f", HouseFeatures()), ("p", _prep(False)),
                                   ("m", RandomForestRegressor(300, min_samples_leaf=3, max_features=0.5,
                                                               n_jobs=-1, random_state=seed))]),
        "gradient_boosting": Pipeline([("f", HouseFeatures()), ("p", _prep(False)),
                                       ("m", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800,
                                                                           max_leaf_nodes=31, min_samples_leaf=20,
                                                                           l2_regularization=0.5, random_state=seed))]),
    }
    models = {k: _log_target(v) for k, v in base.items()}
    # Stacking in log space: a ridge meta-learner over out-of-fold base predictions.
    stack = StackingRegressor([(k, clone(v)) for k, v in base.items()], final_estimator=RidgeCV(), cv=5, n_jobs=1)
    models["stacked_ensemble"] = _log_target(stack)
    return models


class SplitConformal(BaseEstimator, RegressorMixin):
    """Distribution-free prediction intervals via split conformal prediction.

    Residuals are computed in log space and normalised by a difficulty estimate
    (here: a model of |residual|), giving adaptive intervals that are wider for
    hard-to-price homes while keeping the finite-sample coverage guarantee.
    """

    def __init__(self, model, alpha: float = 0.1):
        self.model, self.alpha = model, alpha

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        self.model_ = clone(self.model).fit(X_fit, y_fit)
        fit_res = np.abs(np.log(y_fit) - np.log(self.model_.predict(X_fit)))
        self.scale_ = Pipeline([("f", HouseFeatures()), ("p", _prep(False)),
                                ("m", HistGradientBoostingRegressor(max_iter=200, random_state=0))])
        self.scale_.fit(X_fit, fit_res)
        cal_scores = np.abs(np.log(y_cal) - np.log(self.model_.predict(X_cal))) / self._sigma(X_cal)
        n = len(cal_scores)
        self.q_ = np.quantile(cal_scores, min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n))
        return self

    def _sigma(self, X):
        return np.clip(self.scale_.predict(X), 0.02, None)

    def predict(self, X):
        return self.model_.predict(X)

    def predict_interval(self, X):
        center = np.log(self.model_.predict(X))
        width = self.q_ * self._sigma(X)
        return np.exp(center - width), np.exp(center + width)
