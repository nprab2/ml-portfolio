"""Simulated e-commerce experiment with pre-period data (for CUPED)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_experiment(n_users: int = 80_000, base_cr: float = 0.04, cr_lift: float = 0.08,
                        aov_lift: float = 0.0, seed: int = 4) -> pd.DataFrame:
    """Each user has a latent propensity that drives both pre-period and in-experiment spend."""
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n_users) < 0.5, "B", "A")
    propensity = rng.lognormal(0, 0.8, n_users)
    propensity /= propensity.mean()
    pre_revenue = rng.gamma(0.6, 40, n_users) * propensity          # spend in the 4 weeks before
    cr = np.clip(base_cr * propensity * np.where(group == "B", 1 + cr_lift, 1), 0, 0.9)
    converted = rng.random(n_users) < cr
    aov = rng.lognormal(3.9, 0.7, n_users) * np.where(group == "B", 1 + aov_lift, 1)
    revenue = np.where(converted, aov, 0.0) + 0.15 * pre_revenue * rng.lognormal(0, 0.5, n_users)
    return pd.DataFrame({"user_id": np.arange(n_users), "group": group, "converted": converted.astype(int),
                         "revenue": revenue.round(2), "pre_revenue": pre_revenue.round(2)})
