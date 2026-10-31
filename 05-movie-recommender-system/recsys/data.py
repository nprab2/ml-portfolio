"""Synthetic explicit-ratings data with latent tastes, popularity bias and MNAR exposure."""
from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["action", "comedy", "drama", "sci-fi", "romance", "horror", "documentary", "animation"]


def generate_ratings(n_users: int = 3000, n_items: int = 1500, n_factors: int = 8,
                     avg_ratings_per_user: int = 50, seed: int = 5):
    """Return (ratings, items). Ratings are 1–5 stars.

    Users are more likely to *see* popular items and items matching their taste
    (missing-not-at-random), just like real rating logs.
    """
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, len(GENRES), n_items)
    genre_vecs = rng.normal(0, 1, (len(GENRES), n_factors))
    Q = genre_vecs[item_genre] * 0.8 + rng.normal(0, 0.5, (n_items, n_factors))
    P = rng.normal(0, 1, (n_users, n_factors)) / np.sqrt(n_factors)
    user_bias = rng.normal(0, 0.4, n_users)
    item_quality = rng.normal(0, 0.5, n_items)
    popularity = rng.zipf(1.6, n_items).clip(max=500).astype(float)

    affinity = P @ Q.T / np.sqrt(n_factors) * 2.2
    rows = []
    n_per_user = rng.poisson(avg_ratings_per_user, n_users).clip(5)
    log_pop = np.log(popularity)
    for u in range(n_users):
        exposure = log_pop + 0.8 * affinity[u] + rng.gumbel(size=n_items)  # Gumbel-top-k sampling
        seen = np.argpartition(-exposure, n_per_user[u])[: n_per_user[u]]
        latent = 3.4 + user_bias[u] + item_quality[seen] + affinity[u, seen] + rng.normal(0, 0.6, seen.size)
        rows.append(np.column_stack([np.full(seen.size, u), seen, np.clip(np.rint(latent), 1, 5)]))
    r = np.vstack(rows).astype(int)
    ratings = pd.DataFrame(r, columns=["user_id", "item_id", "rating"])
    ratings["timestamp"] = rng.integers(1_600_000_000, 1_700_000_000, len(ratings))
    items = pd.DataFrame({"item_id": np.arange(n_items), "genre": np.array(GENRES)[item_genre],
                          "popularity_weight": popularity})
    return ratings, items


def temporal_user_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """Per-user split: each user's most recent ``test_frac`` ratings go to test."""
    r = ratings.sort_values(["user_id", "timestamp"])
    rank = r.groupby("user_id").cumcount(ascending=False)
    size = r.groupby("user_id")["item_id"].transform("size")
    is_test = rank < np.ceil(size * test_frac)
    return r[~is_test].reset_index(drop=True), r[is_test].reset_index(drop=True)
