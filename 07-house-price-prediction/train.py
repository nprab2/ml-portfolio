"""House price modelling: base learners vs. stacking, plus conformal prediction intervals.

Usage: python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from housing.data import generate_houses
from housing.model import TARGET, SplitConformal, build_models

RESULTS = Path(__file__).parent / "results"


def metrics(y, p) -> dict:
    return {"rmsle": float(np.sqrt(np.mean((np.log(y) - np.log(p)) ** 2))),
            "mae_usd": float(mean_absolute_error(y, p)),
            "mape": float(mean_absolute_percentage_error(y, p)), "r2": float(r2_score(y, p))}


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    df = generate_houses()
    X, y = df.drop(columns=TARGET), df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    report = {"n_homes": len(df), "median_price": float(np.median(y)), "models": {}}

    fitted = {}
    for name, model in build_models().items():
        fitted[name] = model.fit(X_tr, y_tr)
        report["models"][name] = metrics(y_te, model.predict(X_te))
    stack = fitted["stacked_ensemble"].regressor_
    report["stack_weights"] = dict(zip([n for n, _ in stack.estimators], stack.final_estimator_.coef_.round(3).tolist()))

    # Conformal intervals: hold out a calibration set from the training data.
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.15, random_state=1)
    conf = SplitConformal(build_models()["gradient_boosting"], alpha=0.1).fit(X_fit, y_fit, X_cal, y_cal)
    lo, hi = conf.predict_interval(X_te)
    covered = (y_te >= lo) & (y_te <= hi)
    rel_width = (hi - lo) / conf.predict(X_te)
    big = X_te["sqft"].to_numpy() > 3500
    report["conformal_90"] = {
        "coverage": float(covered.mean()),
        "median_relative_width": float(np.median(rel_width)),
        "coverage_large_homes": float(covered[big].mean()),
        "median_relative_width_large_homes": float(np.median(rel_width[big])),
        "median_relative_width_typical_homes": float(np.median(rel_width[~big])),
    }

    best = fitted["stacked_ensemble"]
    p = best.predict(X_te)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_te / 1e3, p / 1e3, s=3, alpha=0.3)
    lim = [y_te.min() / 1e3, np.percentile(y_te, 99.5) / 1e3]
    axes[0].plot(lim, lim, "r--", lw=1); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Actual price ($k)"); axes[0].set_ylabel("Predicted ($k)")
    axes[0].set_title(f"Stacked ensemble – test R² = {report['models']['stacked_ensemble']['r2']:.3f}")
    pc = conf.predict(X_te)
    order = np.argsort(pc)[:: max(1, len(pc) // 60)]
    axes[1].errorbar(np.arange(order.size), pc[order] / 1e3,
                     yerr=[(pc[order] - lo[order]) / 1e3, (hi[order] - pc[order]) / 1e3], fmt="none", alpha=0.5)
    axes[1].plot(np.arange(order.size), y_te[order] / 1e3, "k.", ms=4, label="actual")
    axes[1].set_title(f"90% conformal intervals (coverage {covered.mean():.1%})"); axes[1].set_ylabel("$k")
    axes[1].set_xlabel("Test homes sorted by prediction"); axes[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "predictions.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    PartialDependenceDisplay.from_estimator(fitted["gradient_boosting"], X_te.sample(1500, random_state=0),
                                            ["sqft", "year_built", "school_rating"], ax=ax)
    fig.suptitle("Partial dependence (gradient boosting)"); fig.tight_layout()
    fig.savefig(RESULTS / "partial_dependence.png", dpi=120); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
