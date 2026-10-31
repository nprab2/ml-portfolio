"""Remaining-useful-life prediction for a turbofan fleet.

Usage: python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdm.data import simulate_fleet
from pdm.features import RUL_CAP, RegimeNormalizer, nasa_score, prognostic_score, window_features

RESULTS = Path(__file__).parent / "results"


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def gbm():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    train = simulate_fleet(200, seed=13)
    test = simulate_fleet(100, seed=14, truncate=True)
    sensors_all = [c for c in train.columns if c.startswith("s") and c[1:].isdigit()]

    norm = RegimeNormalizer().fit(train)
    tr_n, te_n = norm.transform(train), norm.transform(test)
    raw_score = prognostic_score(train, sensors_all)
    norm_score = prognostic_score(tr_n, sensors_all)
    selected = norm_score[norm_score > 0.2].index.tolist()

    Ftr, Fte = window_features(tr_n, selected), window_features(te_n, selected)
    feat_cols = [c for c in Ftr.columns if c not in ("engine_id", "rul")]
    y_tr = np.minimum(Ftr["rul"].to_numpy(), RUL_CAP)
    last = Fte.groupby("engine_id").tail(1)                 # predict at each test engine's last cycle
    y_true = last["rul"].to_numpy()
    y_true_capped = np.minimum(y_true, RUL_CAP)

    models = {
        "mean_baseline": None,
        "ridge_raw_last_cycle": "raw",
        "ridge_regime_normalised_windows": make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(np.logspace(-2, 3, 20))),
        "gradient_boosting_windows": make_pipeline(SimpleImputer(), gbm()),
    }
    report = {"train_engines": 200, "test_engines": 100, "train_rows": len(train),
              "selected_sensors": selected,
              "prognostic_score_raw_top": raw_score.round(3).head(6).to_dict(),
              "prognostic_score_normalised_top": norm_score.round(3).head(6).to_dict(),
              "models": {}}
    preds = {}
    for name, m in models.items():
        if m is None:
            p = np.full(len(last), y_tr.mean())
        elif m == "raw":
            raw_cols = sensors_all + ["setting1", "setting2", "setting3", "cycle"]
            mm = make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 3, 20))).fit(train[raw_cols], np.minimum(train["rul"], RUL_CAP))
            p = mm.predict(test.groupby("engine_id").tail(1)[raw_cols])
        else:
            m.fit(Ftr[feat_cols], y_tr)
            p = m.predict(last[feat_cols])
        p = np.clip(p, 0, None)
        preds[name] = p
        report["models"][name] = {"rmse": rmse(y_true_capped, p), "nasa_score": nasa_score(y_true_capped, p)}

    # Cross-validated RMSE grouped by engine (no engine appears in both train and validation folds).
    cv_pred = cross_val_predict(make_pipeline(SimpleImputer(), gbm()), Ftr[feat_cols], y_tr,
                                groups=Ftr["engine_id"], cv=GroupKFold(5))
    report["gbm_group_cv_rmse_all_cycles"] = rmse(y_tr, cv_pred)

    # Maintenance policy: flag engines predicted to fail within 20 cycles.
    p = preds["gradient_boosting_windows"]
    need, flag = y_true <= 20, p <= 25   # 5-cycle safety margin
    report["maintenance_alert"] = {"engines_truly_within_20_cycles": int(need.sum()),
                                   "recall": float(recall_score(need, flag)), "precision": float(precision_score(need, flag, zero_division=0))}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = np.argsort(y_true_capped)
    axes[0].plot(y_true_capped[order], "k-", lw=2, label="true RUL (capped)")
    axes[0].plot(preds["gradient_boosting_windows"][order], "C0.", label="gradient boosting")
    axes[0].plot(preds["ridge_raw_last_cycle"][order], "C3x", ms=4, alpha=0.6, label="ridge, raw sensors")
    axes[0].set_xlabel("Test engine (sorted by true RUL)"); axes[0].set_ylabel("RUL (cycles)")
    axes[0].set_title("Predicted vs. true remaining useful life"); axes[0].legend()
    eng = train[train.engine_id == 0]
    s = selected[0]
    axes[1].plot(eng["cycle"], eng[s], ".", ms=3, alpha=0.4, label=f"{s} raw")
    axes[1].plot(eng["cycle"], tr_n.loc[eng.index, s] * 2 + eng[s].mean(), "C2-", lw=1,
                 label=f"{s} regime-normalised (scaled)")
    axes[1].set_xlabel("cycle"); axes[1].set_title("Regime normalisation reveals degradation (engine 0)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "rul_predictions.png", dpi=120); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
