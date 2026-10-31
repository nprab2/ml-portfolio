"""Synthetic e-commerce transaction log generated from hidden customer personas."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["electronics", "fashion", "home", "grocery", "beauty"]

# persona: (share, orders/yr, avg basket $, days since last order scale, discount propensity, category weights)
PERSONAS = {
    "champions":       (0.12, 30, 120, 10, 0.15, [3, 2, 2, 2, 2]),
    "loyal_grocery":   (0.25, 40, 45, 6, 0.30, [0.5, 0.5, 1, 6, 1]),
    "big_ticket":      (0.13, 3, 650, 90, 0.10, [7, 1, 3, 0.2, 0.3]),
    "deal_hunters":    (0.22, 12, 55, 30, 0.85, [1, 3, 2, 1, 3]),
    "lapsing":         (0.28, 4, 70, 260, 0.30, [1, 2, 1, 1, 1]),
}


def generate_transactions(n_customers: int = 5000, end: str = "2026-09-30", seed: int = 11):
    """Return (transactions, customers). ``customers`` holds signup date and the hidden
    persona, which is used only to *validate* the clustering, never to fit it."""
    rng = np.random.default_rng(seed)
    names = list(PERSONAS)
    shares = np.array([PERSONAS[p][0] for p in names])
    persona = rng.choice(names, n_customers, p=shares / shares.sum())
    end_ts = pd.Timestamp(end)
    rows, signup = [], []
    for cid, p in enumerate(persona):
        _, freq, basket, recency_scale, disc, cat_w = PERSONAS[p]
        tenure_days = int(rng.uniform(200, 1000))
        last = min(rng.exponential(recency_scale), tenure_days - 1)
        active_days = max(tenure_days - last, 1)
        n_orders = max(1, rng.poisson(freq * rng.lognormal(0, 0.3) * active_days / 365))
        days_ago = np.sort(last + rng.uniform(0, active_days, n_orders))
        days_ago[0] = last
        cw = np.array(cat_w) * rng.lognormal(0, 0.3, len(CATEGORIES))
        cats = rng.choice(CATEGORIES, n_orders, p=cw / cw.sum())
        amounts = basket * rng.lognormal(-0.1, 0.45, n_orders)
        discounted = rng.random(n_orders) < disc
        amounts = np.where(discounted, amounts * 0.75, amounts)
        for d, c, a, dc in zip(days_ago, cats, amounts, discounted):
            rows.append((cid, end_ts - pd.Timedelta(days=float(d)), c, round(float(a), 2), int(dc)))
        signup.append(end_ts - pd.Timedelta(days=tenure_days))
    tx = pd.DataFrame(rows, columns=["customer_id", "timestamp", "category", "amount", "used_discount"])
    customers = pd.DataFrame({"customer_id": np.arange(n_customers), "signup_date": signup,
                              "true_persona": persona})
    return tx, customers
