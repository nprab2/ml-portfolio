"""Synthetic housing market with spatial structure, non-linear effects and heteroscedastic noise."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_houses(n: int = 15_000, n_neighbourhoods: int = 40, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Neighbourhood centres on a 50x50 km map, each with its own price premium.
    centres = rng.uniform(0, 50, (n_neighbourhoods, 2))
    premium = rng.normal(0, 0.25, n_neighbourhoods)
    nb = rng.integers(0, n_neighbourhoods, n)
    xy = centres[nb] + rng.normal(0, 1.5, (n, 2))
    downtown = np.array([25.0, 25.0])
    dist_downtown = np.linalg.norm(xy - downtown, axis=1)

    sqft = np.clip(rng.lognormal(7.45, 0.35, n), 450, 7000)
    beds = np.clip(np.round(sqft / 650 + rng.normal(0, 0.7, n)), 1, 7).astype(int)
    baths = np.clip(np.round(beds * 0.7 + rng.normal(0, 0.5, n) * 2) / 2, 1, 5)
    year_built = np.clip(rng.normal(1985, 22, n), 1900, 2025).astype(int)
    lot_sqft = np.clip(sqft * rng.lognormal(1.0, 0.5, n) * (1 + dist_downtown / 25), 800, 80_000)
    condition = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.45, 0.25, 0.10])
    garage = rng.choice([0, 1, 2, 3], n, p=[0.15, 0.35, 0.42, 0.08])
    has_pool = (rng.random(n) < 0.12 + 0.1 * (sqft > 3000)).astype(int)
    renovated = (rng.random(n) < np.where(year_built < 1970, 0.4, 0.1)).astype(int)
    school_rating = np.clip(np.round(6 + premium[nb] * 8 + rng.normal(0, 1.2, n)), 1, 10).astype(int)
    property_type = rng.choice(["single_family", "townhouse", "condo"], n, p=[0.65, 0.2, 0.15])

    age = 2026 - year_built
    log_price = (
        7.55
        + 0.75 * np.log(sqft)
        - 0.018 * dist_downtown                                   # location gradient
        + premium[nb]                                             # neighbourhood premium
        + 0.04 * school_rating
        + 0.07 * (condition - 3)
        - 0.004 * age + 0.000035 * age ** 2                       # U-shape: historic homes recover value
        + 0.12 * renovated * (age > 50)
        + 0.03 * garage + 0.05 * has_pool
        + 0.08 * np.log(lot_sqft / sqft)
        - 0.12 * (property_type == "condo") - 0.06 * (property_type == "townhouse")
        + 0.02 * baths
    )
    noise = rng.normal(0, 0.08 + 0.05 * (sqft > 3500), n)          # luxury homes are harder to price
    price = np.round(np.exp(log_price + noise) / 1000) * 1000

    df = pd.DataFrame({
        "price": price, "sqft": sqft.round(), "beds": beds, "baths": baths, "year_built": year_built,
        "lot_sqft": lot_sqft.round(), "condition": condition, "garage_spaces": garage, "has_pool": has_pool,
        "renovated": renovated, "school_rating": school_rating, "property_type": property_type,
        "neighbourhood": [f"NB{k:02d}" for k in nb], "x_km": xy[:, 0].round(3), "y_km": xy[:, 1].round(3),
    })
    # Store numerics as float (listings data has NaNs; also required for partial dependence).
    ints = df.select_dtypes("integer").columns
    df[ints] = df[ints].astype(float)
    # Missing values typical of listings data.
    for col, rate in [("lot_sqft", 0.05), ("year_built", 0.02), ("garage_spaces", 0.03)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    return df
