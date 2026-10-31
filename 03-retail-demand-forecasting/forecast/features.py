"""Horizon-safe feature engineering and evaluation metrics for direct forecasting.

All target-derived features are shifted by at least ``HORIZON`` days, so a single
global model can forecast every day of the next ``HORIZON`` days without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 28
KEYS = ["store_id", "item_id"]


def make_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = df.sort_values(KEYS + ["date"]).copy()
    g = df.groupby(KEYS, sort=False)["sales"]
    for lag in [horizon, horizon + 7, horizon + 14, 364]:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(horizon)
    sg = shifted.groupby([df[k] for k in KEYS], sort=False)
    for w in [7, 28, 91]:
        df[f"roll_mean_{w}"] = sg.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["roll_std_28"] = sg.transform(lambda s: s.rolling(28, min_periods=7).std())
    # Same-weekday average over the last 4 available weeks.
    df["dow_mean_4w"] = df[[f"lag_{horizon}", f"lag_{horizon + 7}", f"lag_{horizon + 14}"]].mean(axis=1)

    # Known-in-advance covariates.
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["price_ratio"] = df["price"] / df.groupby(KEYS)["price"].transform("max")
    df["log_price"] = np.log(df["price"])
    return df


FEATURES = ["store_id", "item_id", "lag_28", "lag_35", "lag_42", "lag_364", "roll_mean_7",
            "roll_mean_28", "roll_mean_91", "roll_std_28", "dow_mean_4w", "dow", "month",
            "doy_sin", "doy_cos", "price_ratio", "log_price", "on_promo", "is_holiday"]
CATEGORICAL = ["store_id", "item_id", "dow"]


def wape(y, yhat) -> float:
    """Weighted absolute percentage error – robust to zero sales, used widely in retail."""
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def bias(y, yhat) -> float:
    return float((yhat - y).sum() / y.sum())


def coverage(y, lo, hi) -> float:
    return float(((y >= lo) & (y <= hi)).mean())
