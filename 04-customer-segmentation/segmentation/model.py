"""RFM+ feature engineering, model selection and cluster-stability analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import CATEGORIES

MODEL_FEATURES = ["recency_days", "frequency_per_year", "avg_order_value", "monetary_per_year",
                  "discount_share", "tenure_days", "category_entropy"] + [f"share_{c}" for c in CATEGORIES]
PRIOR_STRENGTH = 3.0
LOG_FEATURES = ["recency_days", "frequency_per_year", "avg_order_value", "monetary_per_year"]


def build_customer_features(tx: pd.DataFrame, customers: pd.DataFrame, as_of: str) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    g = tx.groupby("customer_id")
    f = pd.DataFrame({
        "recency_days": (as_of - g["timestamp"].max()).dt.days,
        "n_orders": g.size(),
        "total_spend": g["amount"].sum(),
        "avg_order_value": g["amount"].mean(),
        "discount_share": g["used_discount"].mean(),
    })
    f = f.join(customers.set_index("customer_id")[["signup_date"]])
    f["tenure_days"] = (as_of - f.pop("signup_date")).dt.days
    years = np.maximum(f["tenure_days"], 30) / 365
    f["frequency_per_year"] = f["n_orders"] / years
    f["monetary_per_year"] = f["total_spend"] / years
    # Empirical-Bayes smoothing: customers with 1-2 orders would otherwise get extreme 0/1
    # shares that create spurious micro-clusters. Shrink toward the population mix.
    counts = pd.crosstab(tx["customer_id"], tx["category"]).reindex(columns=CATEGORIES, fill_value=0)
    prior = counts.sum() / counts.to_numpy().sum()
    n = counts.sum(axis=1).to_numpy()[:, None]
    shares = (counts + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH)
    for c in CATEGORIES:
        f[f"share_{c}"] = shares[c]
    disc_prior = tx["used_discount"].mean()
    f["discount_share"] = (g["used_discount"].sum() + PRIOR_STRENGTH * disc_prior) / (f["n_orders"] + PRIOR_STRENGTH)
    p = shares.to_numpy().clip(1e-12)
    f["category_entropy"] = -(p * np.log(p)).sum(axis=1) / np.log(len(CATEGORIES))
    return f


def to_model_matrix(f: pd.DataFrame) -> np.ndarray:
    X = f[MODEL_FEATURES].copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])   # RFM values are heavily right-skewed
    return StandardScaler().fit_transform(X)


def select_k(X: np.ndarray, k_range=range(2, 11), seed: int = 0, sample: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=seed).fit(X)
        gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X[idx], km.labels_[idx]),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_),
                     "gmm_bic": gmm.bic(X)})
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, k: int, n_boot: int = 20, seed: int = 0) -> float:
    """Mean ARI between the full-data clustering and clusterings fit on bootstrap resamples
    (compared on all points). Values > 0.8 indicate stable, trustworthy segments."""
    ref = KMeans(k, n_init=10, random_state=seed).fit(X).labels_
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        km = KMeans(k, n_init=5, random_state=b).fit(X[idx])
        scores.append(adjusted_rand_score(ref, km.predict(X)))
    return float(np.mean(scores))


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Rule-based, human-readable names from each cluster's profile (relative to the median)."""
    med = profile.median()
    names = {}
    for c, r in profile.iterrows():
        if r["avg_order_value"] > 3 * med["avg_order_value"]:
            n = "Big-ticket buyers"
        elif r["recency_days"] > 3 * med["recency_days"]:
            n = "At-risk / lapsing"
        elif r["discount_share"] > 0.6:
            n = "Deal hunters"
        elif r["share_grocery"] > 0.5:
            n = "Loyal everyday shoppers"
        elif r["monetary_per_year"] > 1.5 * med["monetary_per_year"]:
            n = "Champions"
        else:
            n = "Regulars"
        names[c] = n
    return names
