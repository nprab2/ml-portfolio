"""End-to-end churn experiment: data -> models -> evaluation -> business threshold.

Usage:  python train.py [--n 20000] [--seed 42]
Outputs: results/metrics.json and figures in results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn.data import generate_churn_data
from churn.model import (TARGET, build_baseline, build_gbm, expected_profit,
                         optimal_threshold)

RESULTS = Path(__file__).parent / "results"


def main(n: int, seed: int) -> dict:
    RESULTS.mkdir(exist_ok=True)
    df = generate_churn_data(n, seed)
    X, y = df.drop(columns=[TARGET, "customer_id"]), df[TARGET].values

    # 60 / 20 / 20 split: threshold is tuned on validation, reported on untouched test.
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    models = {"logistic_regression": build_baseline(), "gradient_boosting": build_gbm()}
    report: dict = {"n_customers": n, "churn_rate": float(y.mean()), "models": {}}

    plt.figure(figsize=(6, 5))
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]
        report["models"][name] = {
            "cv_roc_auc_mean": float(cv_auc.mean()),
            "cv_roc_auc_std": float(cv_auc.std()),
            "test_roc_auc": float(roc_auc_score(y_test, p_test)),
            "test_pr_auc": float(average_precision_score(y_test, p_test)),
            "test_brier": float(brier_score_loss(y_test, p_test)),
        }
        fpr, tpr, _ = roc_curve(y_test, p_test)
        plt.plot(fpr, tpr, label=f"{name} (AUC={report['models'][name]['test_roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC – test set")
    plt.legend(); plt.tight_layout(); plt.savefig(RESULTS / "roc_curve.png", dpi=120); plt.close()

    # Business threshold on the champion model.
    champion_name = max(report["models"], key=lambda m: report["models"][m]["cv_roc_auc_mean"])
    report["champion"] = champion_name
    champion = models[champion_name]
    p_val = champion.predict_proba(X_val)[:, 1]
    p_test = champion.predict_proba(X_test)[:, 1]
    thr, _ = optimal_threshold(y_val, p_val)
    profit_opt = expected_profit(y_test, p_test, thr)
    profit_05 = expected_profit(y_test, p_test, 0.5)
    profit_all = expected_profit(y_test, p_test, 0.0)
    report["business"] = {
        "optimal_threshold": thr,
        "test_profit_at_optimal": profit_opt,
        "test_profit_at_0.5": profit_05,
        "test_profit_target_everyone": profit_all,
        "test_f1_at_optimal": float(f1_score(y_test, p_test >= thr)),
        "customers_targeted_pct": float((p_test >= thr).mean()),
    }

    grid = np.linspace(0.01, 0.99, 99)
    plt.figure(figsize=(6, 4))
    plt.plot(grid, [expected_profit(y_test, p_test, t) for t in grid])
    plt.axvline(thr, color="r", ls="--", label=f"chosen threshold={thr:.2f}")
    plt.xlabel("Decision threshold"); plt.ylabel("Expected profit ($)")
    plt.title("Retention campaign profit vs threshold"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS / "profit_curve.png", dpi=120); plt.close()

    # Model-agnostic explanation.
    imp = permutation_importance(champion, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed)
    order = np.argsort(imp.importances_mean)[::-1]
    report["permutation_importance"] = {X_test.columns[i]: float(imp.importances_mean[i]) for i in order}
    plt.figure(figsize=(6, 4))
    top = order[:10][::-1]
    plt.barh(X_test.columns[top], imp.importances_mean[top], xerr=imp.importances_std[top])
    plt.xlabel("Drop in ROC-AUC when shuffled"); plt.title("Permutation importance")
    plt.tight_layout(); plt.savefig(RESULTS / "feature_importance.png", dpi=120); plt.close()

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(json.dumps(main(args.n, args.seed), indent=2))
