"""Train and compare recommenders on rating prediction and top-N ranking.

Usage: python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recsys.data import generate_ratings, temporal_user_split
from recsys.models import (BiasBaseline, BiasedALS, ImplicitALS, ItemKNN, Popularity,
                           ranking_metrics)

RESULTS = Path(__file__).parent / "results"


def rmse(model, df):
    pred = model.predict(df["user_id"].to_numpy(), df["item_id"].to_numpy())
    return float(np.sqrt(np.mean((pred - df["rating"].to_numpy()) ** 2)))


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    ratings, items = generate_ratings()
    nu, ni = ratings["user_id"].max() + 1, len(items)
    train, test = temporal_user_split(ratings, 0.2)
    sub_train, valid = temporal_user_split(train, 0.2)

    # Hyper-parameter search for ALS on a temporal validation split (test set untouched).
    grid = []
    for k in [16, 32, 64]:
        for lam in [0.05, 0.1, 0.2]:
            m = BiasedALS(n_factors=k, factor_reg=lam, n_iter=8).fit(sub_train, nu, ni)
            grid.append({"n_factors": k, "factor_reg": lam, "valid_rmse": rmse(m, valid)})
    best = min(grid, key=lambda g: g["valid_rmse"])

    # Implicit ALS is tuned for ranking (NDCG@10) rather than RMSE.
    igrid = []
    for k in [16, 32]:
        for reg, alpha in [(30.0, 10.0), (100.0, 10.0), (100.0, 20.0), (300.0, 20.0)]:
            m = ImplicitALS(n_factors=k, reg=reg, alpha=alpha, n_iter=8).fit(sub_train, nu, ni)
            igrid.append({"n_factors": k, "reg": reg, "alpha": alpha,
                          "valid_ndcg@10": ranking_metrics(m, sub_train, valid, ni)["ndcg@10"]})
    ibest = max(igrid, key=lambda g: g["valid_ndcg@10"])

    global_mean = float(train["rating"].mean())
    bias = BiasBaseline().fit(train, nu, ni)
    als = BiasedALS(n_factors=best["n_factors"], factor_reg=best["factor_reg"]).fit(train, nu, ni)
    knn = ItemKNN().fit(train, nu, ni)
    pop = Popularity().fit(train, nu, ni)
    ials = ImplicitALS(n_factors=ibest["n_factors"], reg=ibest["reg"], alpha=ibest["alpha"]).fit(train, nu, ni)

    report = {
        "n_users": int(nu), "n_items": int(ni), "n_ratings": len(ratings),
        "density": len(ratings) / (nu * ni), "als_grid": grid, "als_best": best,
        "implicit_grid": igrid, "implicit_best": ibest,
        "rmse": {"global_mean": float(np.sqrt(np.mean((test["rating"] - global_mean) ** 2))),
                 "bias_baseline": rmse(bias, test), "biased_als": rmse(als, test)},
        "ranking": {"popularity": ranking_metrics(pop, train, test, ni),
                    "item_knn": ranking_metrics(knn, train, test, ni),
                    "biased_als (rating-optimised)": ranking_metrics(als, train, test, ni),
                    "implicit_als (ranking-optimised)": ranking_metrics(ials, train, test, ni)},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(als.history) + 1), als.history, "o-")
    axes[0].set_xlabel("ALS iteration"); axes[0].set_ylabel("Train residual RMSE"); axes[0].set_title("ALS convergence")
    names = list(report["ranking"])
    x = np.arange(4)
    for j, n in enumerate(names):
        vals = [report["ranking"][n][m] for m in ["precision@10", "recall@10", "ndcg@10", "catalog_coverage"]]
        axes[1].bar(x + j * 0.2, vals, 0.2, label=n)
    axes[1].set_xticks(x + 0.3, ["Precision@10", "Recall@10", "NDCG@10", "Coverage"])
    axes[1].set_title("Top-10 recommendation quality (test)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "recommender_comparison.png", dpi=120); plt.close(fig)

    # Example recommendations for one user, with genre context.
    u = 7
    scores = ials.score_all(np.array([u]))[0]
    scores[train.loc[train.user_id == u, "item_id"]] = -np.inf
    liked = train[(train.user_id == u) & (train.rating >= 4)].merge(items, on="item_id")["genre"].value_counts()
    report["example_user"] = {"user_id": u, "liked_genres_in_history": liked.head(3).to_dict(),
                              "top10_genres": items.loc[np.argsort(-scores)[:10], "genre"].value_counts().to_dict()}

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
