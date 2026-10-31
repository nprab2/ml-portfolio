"""Synthetic card-transaction stream with customer behaviour and injected fraud."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "gas", "restaurant", "online_retail", "travel", "electronics", "pharmacy"]
CAT_RISK = {"grocery": 0.2, "gas": 0.5, "restaurant": 0.3, "online_retail": 1.6,
            "travel": 1.2, "electronics": 1.8, "pharmacy": 0.2}


def generate_transactions(n_customers: int = 3000, days: int = 90, fraud_rate: float = 0.006,
                          seed: int = 7) -> pd.DataFrame:
    """Simulate ``days`` of transactions. Fraud events differ in amount, time,
    merchant mix, distance and velocity – but overlap with legitimate behaviour."""
    rng = np.random.default_rng(seed)
    spend_level = rng.lognormal(3.4, 0.6, n_customers)          # typical ticket size per customer
    daily_rate = rng.gamma(2.0, 0.9, n_customers)               # transactions per day
    home_lat, home_lon = rng.uniform(25, 48, n_customers), rng.uniform(-122, -71, n_customers)

    n_tx = rng.poisson(daily_rate * days)
    cust = np.repeat(np.arange(n_customers), n_tx)
    n = cust.size
    ts = rng.uniform(0, days * 24 * 3600, n)
    hour = (ts / 3600) % 24
    # Legit activity is mostly daytime: resample night-time hours with high probability.
    night = (hour < 6) & (rng.random(n) < 0.8)
    ts[night] += rng.uniform(8, 14, night.sum()) * 3600

    is_fraud = rng.random(n) < fraud_rate
    cat_p = np.array([0.30, 0.15, 0.2, 0.15, 0.04, 0.04, 0.12])
    category = rng.choice(CATEGORIES, n, p=cat_p)
    fraud_cat_p = np.array([CAT_RISK[c] for c in CATEGORIES]); fraud_cat_p /= fraud_cat_p.sum()
    category[is_fraud] = rng.choice(CATEGORIES, is_fraud.sum(), p=fraud_cat_p)

    amount = spend_level[cust] * rng.lognormal(0, 0.5, n)
    amount[is_fraud] *= rng.lognormal(1.0, 0.8, is_fraud.sum())
    distance_km = rng.exponential(8, n)
    distance_km[is_fraud] = np.where(rng.random(is_fraud.sum()) < 0.6,
                                     rng.exponential(600, is_fraud.sum()), rng.exponential(10, is_fraud.sum()))
    card_present = rng.random(n) < np.where(np.isin(category, ["online_retail"]), 0.0, 0.85)
    card_present[is_fraud] &= rng.random(is_fraud.sum()) < 0.3
    new_device = rng.random(n) < np.where(is_fraud, 0.45, 0.03)

    # Fraud often comes in bursts: shift some fraud to the night-time and add follow-up probes.
    fraud_idx = np.flatnonzero(is_fraud)
    late = fraud_idx[rng.random(fraud_idx.size) < 0.5]
    ts[late] = np.floor(ts[late] / 86400) * 86400 + rng.uniform(0, 5, late.size) * 3600

    df = pd.DataFrame({
        "customer_id": cust, "timestamp": ts, "amount": np.round(amount, 2),
        "merchant_category": category, "distance_from_home_km": np.round(distance_km, 1),
        "card_present": card_present.astype(int), "new_device": new_device.astype(int),
        "is_fraud": is_fraud.astype(int),
    })
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer history features using only *past* transactions (no look-ahead)."""
    df = df.sort_values(["customer_id", "timestamp"], kind="stable").copy()
    g = df.groupby("customer_id", sort=False)
    df["hour"] = (df["timestamp"] / 3600 % 24).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    past_mean = g["amount"].transform(lambda s: s.shift().expanding().mean())
    past_std = g["amount"].transform(lambda s: s.shift().expanding().std())
    df["amount_vs_customer_mean"] = df["amount"] / past_mean
    df["amount_zscore"] = (df["amount"] - past_mean) / past_std.replace(0, np.nan)
    df["seconds_since_last_tx"] = g["timestamp"].diff()
    df["customer_tx_count"] = g.cumcount()

    # Rolling velocity: transactions and spend in the previous hour (time-based window).
    tmp = df[["customer_id", "timestamp", "amount"]].copy()
    tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="s")
    rolled = (tmp.set_index("dt").groupby("customer_id", sort=False)["amount"]
                 .rolling("1h", closed="left").agg(["count", "sum"]))
    df["tx_last_1h"] = np.nan_to_num(rolled["count"].to_numpy())  # empty window -> 0
    df["spend_last_1h"] = np.nan_to_num(rolled["sum"].to_numpy())
    df["log_amount"] = np.log1p(df["amount"])
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
