"""Models and fraud-specific evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["log_amount", "amount_vs_customer_mean", "amount_zscore", "distance_from_home_km",
           "card_present", "new_device", "hour", "is_night", "seconds_since_last_tx",
           "customer_tx_count", "tx_last_1h", "spend_last_1h"]
CATEGORICAL = ["merchant_category"]
TARGET = "is_fraud"


def _prep(scale: bool, numeric: list[str]) -> ColumnTransformer:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def build_models(seed: int = 0, numeric: list[str] | None = None) -> dict[str, Pipeline]:
    numeric = list(NUMERIC if numeric is None else numeric)
    return {
        "isolation_forest (unsupervised)": Pipeline([
            ("prep", _prep(True, numeric)),
            ("clf", IsolationForest(n_estimators=300, contamination="auto", random_state=seed)),
        ]),
        "logistic_regression (balanced)": Pipeline([
            ("prep", _prep(True, numeric)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", _prep(False, numeric)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.06, max_iter=500, max_leaf_nodes=31,
                                                   min_samples_leaf=50, l2_regularization=1.0,
                                                   class_weight={0: 1, 1: 10}, early_stopping=True,
                                                   random_state=seed)),
        ]),
    }


def score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more suspicious, regardless of model family."""
    if isinstance(model.named_steps["clf"], IsolationForest):
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    top = np.argsort(-s)[:k]
    return float(y[top].mean())


def recall_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.01) -> float:
    """Fraction of fraud caught while flagging at most ``max_fpr`` of legit traffic."""
    thr = np.quantile(s[y == 0], 1 - max_fpr)
    return float((s[y == 1] > thr).mean())


def dollars_saved(y, s, amounts, review_budget: int, review_cost: float = 5.0) -> float:
    """Fraud $ intercepted by reviewing the top ``review_budget`` alerts, minus review cost."""
    top = np.argsort(-s)[:review_budget]
    return float(amounts[top][y[top] == 1].sum() - review_budget * review_cost)
