"""Customer segmentation: features -> choose k -> cluster -> validate -> profile.

Usage: python train.py [--customers 5000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from segmentation.data import generate_transactions
from segmentation.model import (bootstrap_stability, build_customer_features, name_segments,
                                select_k, to_model_matrix)

RESULTS = Path(__file__).parent / "results"
AS_OF = "2026-09-30"


def main(n_customers: int, seed: int) -> dict:
    RESULTS.mkdir(exist_ok=True)
    tx, customers = generate_transactions(n_customers, end=AS_OF, seed=seed)
    feats = build_customer_features(tx, customers, AS_OF)
    X = to_model_matrix(feats)

    scores = select_k(X, seed=seed)
    scores.to_csv(RESULTS / "k_selection.csv", index=False)
    k = int(scores.loc[scores["silhouette"].idxmax(), "k"])

    km = KMeans(k, n_init=20, random_state=seed).fit(X)
    feats["segment"] = km.labels_
    profile_cols = ["recency_days", "frequency_per_year", "avg_order_value", "monetary_per_year",
                    "discount_share", "share_grocery", "share_electronics"]
    profile = feats.groupby("segment")[profile_cols].median()
    names = name_segments(profile)
    profile.insert(0, "name", profile.index.map(names))
    profile.insert(1, "customers", feats["segment"].value_counts().sort_index())
    profile.insert(2, "revenue_share", feats.groupby("segment")["total_spend"].sum() / feats["total_spend"].sum())
    profile.round(3).to_csv(RESULTS / "segment_profiles.csv")

    truth = customers.set_index("customer_id").loc[feats.index, "true_persona"]
    report = {
        "n_customers": n_customers, "n_transactions": len(tx), "chosen_k": k,
        "k_selection": scores.round(4).to_dict(orient="records"),
        "bootstrap_stability_ari": bootstrap_stability(X, k, seed=seed),
        "ari_vs_hidden_personas": float(adjusted_rand_score(truth, km.labels_)),
        "segments": profile.round(3).reset_index().to_dict(orient="records"),
        "confusion_vs_personas": pd.crosstab(truth, feats["segment"].map(names)).to_dict(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(scores["k"], scores["silhouette"], "o-", label="silhouette (↑)")
    ax2 = axes[0].twinx(); ax2.plot(scores["k"], scores["davies_bouldin"], "s--C1", label="Davies–Bouldin (↓)")
    axes[0].axvline(k, color="grey", ls=":"); axes[0].set_xlabel("k"); axes[0].set_title("Choosing the number of segments")
    axes[0].legend(loc="upper left"); ax2.legend(loc="upper right")
    pcs = PCA(2, random_state=seed).fit_transform(X)
    for c in range(k):
        m = km.labels_ == c
        axes[1].scatter(pcs[m, 0], pcs[m, 1], s=4, alpha=0.5, label=f"{c}: {names[c]}")
    axes[1].set_title("Segments in PCA space"); axes[1].legend(markerscale=3, fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "segments.png", dpi=120); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    r = main(a.customers, a.seed)
    print(json.dumps({k: v for k, v in r.items() if k != "k_selection"}, indent=2, default=str))
