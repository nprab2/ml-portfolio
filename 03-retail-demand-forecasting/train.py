"""Rolling-origin backtest of a global gradient-boosting forecaster vs. baselines.

Usage: python train.py [--folds 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast.data import generate_sales
from forecast.features import (CATEGORICAL, FEATURES, HORIZON, bias, coverage, make_features,
                               wape)

RESULTS = Path(__file__).parent / "results"


def gbm(loss: str = "poisson", quantile: float | None = None) -> HistGradientBoostingRegressor:
    kw = dict(loss=loss, learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=50,
              categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=0)
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def backtest(df: pd.DataFrame, n_folds: int) -> tuple[dict, pd.DataFrame]:
    last = df["date"].max()
    origins = [last - pd.Timedelta(days=HORIZON * k - 1) for k in range(n_folds, 0, -1)]
    rows, preds = [], []
    for origin in origins:
        train = df[(df["date"] < origin) & df["lag_364"].notna()]
        test = df[(df["date"] >= origin) & (df["date"] < origin + pd.Timedelta(days=HORIZON))]
        Xtr, ytr, Xte, yte = train[FEATURES], train["sales"], test[FEATURES], test["sales"].to_numpy()

        point = gbm().fit(Xtr, ytr).predict(Xte)
        lo = gbm("quantile", 0.1).fit(Xtr, ytr).predict(Xte)
        hi = gbm("quantile", 0.9).fit(Xtr, ytr).predict(Xte)
        naive = test["lag_28"].to_numpy()                 # seasonal naive (same weekday 4 weeks ago)
        ma = test["roll_mean_28"].to_numpy()               # trailing 28-day mean
        rows.append({"origin": str(origin.date()),
                     "wape_gbm": wape(yte, point), "wape_seasonal_naive": wape(yte, naive),
                     "wape_moving_avg": wape(yte, ma), "bias_gbm": bias(yte, point),
                     "p10_p90_coverage": coverage(yte, lo, hi)})
        preds.append(test[["date", "store_id", "item_id", "sales"]].assign(gbm=point, lo=lo, hi=hi, naive=naive))
    return {"folds": rows}, pd.concat(preds)


def main(n_folds: int) -> dict:
    RESULTS.mkdir(exist_ok=True)
    df = make_features(generate_sales())
    report, preds = backtest(df, n_folds)
    folds = pd.DataFrame(report["folds"])
    report["mean"] = folds.drop(columns="origin").mean().to_dict()
    report["improvement_vs_seasonal_naive"] = 1 - report["mean"]["wape_gbm"] / report["mean"]["wape_seasonal_naive"]
    report["n_series"] = int(df.groupby(["store_id", "item_id"]).ngroups)

    # Total-chain view for the last fold, plus one example series with intervals.
    agg = preds.groupby("date")[["sales", "gbm", "naive"]].sum()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    agg.plot(ax=axes[0], style=["k-", "C0-", "C1--"])
    axes[0].set_title("Total units across all stores (backtest folds)"); axes[0].set_xlabel("")
    one = preds[(preds.store_id == 0) & (preds.item_id == 3)].set_index("date")
    axes[1].fill_between(one.index, one["lo"], one["hi"], alpha=0.25, label="P10–P90 interval")
    axes[1].plot(one.index, one["sales"], "k.", label="actual")
    axes[1].plot(one.index, one["gbm"], "C0-", label="forecast")
    axes[1].set_title("Store 0 / item 3 with probabilistic forecast"); axes[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "forecast.png", dpi=120); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    print(json.dumps(main(ap.parse_args().folds), indent=2))
