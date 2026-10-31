"""Feature engineering, modelling and business-aware thresholding for churn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["contract", "internet_service", "payment_method"]
NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges", "num_products",
    "support_calls_90d", "is_senior", "paperless_billing", "has_partner",
    "charges_per_product", "calls_per_tenure_year", "is_new_customer",
]
TARGET = "churned"


class ChurnFeatures(BaseEstimator, TransformerMixin):
    """Stateless domain features. Kept inside the pipeline to avoid train/serve skew."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["total_charges"] = X["total_charges"].fillna(X["monthly_charges"] * X["tenure_months"])
        X["charges_per_product"] = X["monthly_charges"] / X["num_products"]
        X["calls_per_tenure_year"] = X["support_calls_90d"] / (X["tenure_months"] / 12 + 1)
        X["is_new_customer"] = (X["tenure_months"] <= 6).astype(int)
        return X


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_baseline() -> Pipeline:
    return Pipeline(
        [
            ("features", ChurnFeatures()),
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def build_gbm(calibrate: bool = True, random_state: int = 42) -> Pipeline:
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    clf = CalibratedClassifierCV(gbm, method="isotonic", cv=3) if calibrate else gbm
    return Pipeline(
        [
            ("features", ChurnFeatures()),
            ("prep", _preprocessor(scale=False)),
            ("clf", clf),
        ]
    )


def expected_profit(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    offer_cost: float = 50.0,
    customer_value: float = 600.0,
    save_rate: float = 0.35,
) -> float:
    """Profit of sending a retention offer to everyone scored above ``threshold``.

    Each targeted customer costs ``offer_cost``. A targeted true churner is retained
    with probability ``save_rate``, recovering ``customer_value``.
    """
    targeted = proba >= threshold
    true_pos = np.sum(targeted & (y_true == 1))
    return float(true_pos * save_rate * customer_value - targeted.sum() * offer_cost)


def optimal_threshold(y_true, proba, **profit_kwargs) -> tuple[float, float]:
    """Grid-search the decision threshold that maximises expected profit."""
    grid = np.linspace(0.01, 0.99, 99)
    profits = np.array([expected_profit(y_true, proba, t, **profit_kwargs) for t in grid])
    best = int(np.argmax(profits))
    return float(grid[best]), float(profits[best])
