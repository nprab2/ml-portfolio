"""Fraud detection experiment with an out-of-time test split.

Usage: python train.py [--customers 3000] [--days 90]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraud.data import add_behavioural_features, generate_transactions
from fraud.model import (CATEGORICAL, NUMERIC, TARGET, build_models, dollars_saved,
                         precision_at_k, recall_at_fpr, score)

RESULTS = Path(__file__).parent / "results"


def main(customers: int, days: int, seed: int) -> dict:
    RESULTS.mkdir(exist_ok=True)
    df = add_behavioural_features(generate_transactions(customers, days, seed=seed))

    # Out-of-time split: train on the first 70% of the period, test on the rest.
    cutoff = df["timestamp"].quantile(0.7)
    train, test = df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]
    feats = NUMERIC + CATEGORICAL
    X_tr, y_tr = train[feats], train[TARGET].to_numpy()
    X_te, y_te = test[feats], test[TARGET].to_numpy()
    amounts = test["amount"].to_numpy()
    budget = int(round(len(test) * 0.005))  # analysts can review 0.5% of transactions

    report = {"n_transactions": len(df), "fraud_rate": float(df[TARGET].mean()),
              "test_transactions": len(test), "review_budget": budget,
              "fraud_dollars_in_test": float(amounts[y_te == 1].sum()), "models": {}}

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in build_models(seed).items():
        model.fit(X_tr, y_tr)
        s = score(model, X_te)
        report["models"][name] = {
            "pr_auc": float(average_precision_score(y_te, s)),
            "roc_auc": float(roc_auc_score(y_te, s)),
            "recall_at_1pct_fpr": recall_at_fpr(y_te, s, 0.01),
            f"precision_at_top_{budget}": precision_at_k(y_te, s, budget),
            "net_dollars_saved": dollars_saved(y_te, s, amounts, budget),
        }
        p, r, _ = precision_recall_curve(y_te, s)
        ax.plot(r, p, label=f"{name} (AP={report['models'][name]['pr_auc']:.3f})")
    ax.axhline(y_te.mean(), color="k", ls="--", lw=0.8, label="random")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–recall (out-of-time test)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULTS / "pr_curve.png", dpi=120); plt.close(fig)

    # Ablation: how much do the behavioural (history-based) features add?
    behavioural = {"amount_vs_customer_mean", "amount_zscore", "seconds_since_last_tx",
                   "customer_tx_count", "tx_last_1h", "spend_last_1h"}
    reduced = [c for c in NUMERIC if c not in behavioural]
    m = build_models(seed, numeric=reduced)["gradient_boosting"].fit(train[reduced + CATEGORICAL], y_tr)
    report["ablation_no_behavioural_pr_auc"] = float(
        average_precision_score(y_te, score(m, test[reduced + CATEGORICAL])))

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=3000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(json.dumps(main(a.customers, a.days, a.seed), indent=2))
