"""Synthetic multi-store, multi-item daily sales with realistic structure."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS = ["01-01", "05-27", "07-04", "09-02", "11-28", "12-24", "12-25"]


def generate_sales(n_stores: int = 10, n_items: int = 20, start: str = "2022-01-01",
                   periods: int = 1095, seed: int = 3) -> pd.DataFrame:
    """Daily unit sales with trend, weekly/yearly seasonality, holidays, promotions,
    price elasticity and negative-binomial noise (count data, overdispersed)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="D")
    t = np.arange(periods)
    dow, doy = dates.dayofweek.to_numpy(), dates.dayofyear.to_numpy()
    is_holiday = dates.strftime("%m-%d").isin(HOLIDAYS)

    frames = []
    store_size = rng.lognormal(0, 0.35, n_stores)
    for i in range(n_items):
        base = rng.lognormal(2.3, 0.7)
        weekly_amp = rng.uniform(0.05, 0.35)
        yearly_amp = rng.uniform(0.0, 0.5)
        yearly_phase = rng.uniform(0, 2 * np.pi)
        trend = rng.normal(0.0004, 0.0004)
        elasticity = rng.uniform(-2.5, -0.8)
        base_price = round(rng.uniform(2, 30), 2)
        for s in range(n_stores):
            promo = rng.random(periods) < 0.06
            # Promotions last a few days: dilate single-day flags.
            promo = np.convolve(promo, np.ones(4), mode="same") > 0
            price = base_price * np.where(promo, rng.uniform(0.7, 0.85), 1.0)
            level = (base * store_size[s] * np.exp(trend * t)
                     * (1 + weekly_amp * np.isin(dow, [4, 5]) - 0.5 * weekly_amp * (dow == 0))
                     * (1 + yearly_amp * np.sin(2 * np.pi * doy / 365.25 + yearly_phase))
                     * (price / base_price) ** elasticity
                     * np.where(is_holiday, rng.uniform(1.3, 2.0), 1.0))
            r = 5.0  # NB dispersion
            sales = rng.negative_binomial(r, r / (r + level))
            frames.append(pd.DataFrame({"date": dates, "store_id": s, "item_id": i,
                                        "price": np.round(price, 2), "on_promo": promo.astype(int),
                                        "is_holiday": is_holiday.astype(int), "sales": sales}))
    return pd.concat(frames, ignore_index=True)
