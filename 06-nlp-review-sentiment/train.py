"""Train sentiment models, evaluate on a held-out set and on linguistic challenge sets.

Usage: python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sentiment.data import challenge_sets, generate_reviews
from sentiment.model import build_models, top_features

RESULTS = Path(__file__).parent / "results"


def evaluate(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {"accuracy": float(accuracy_score(y, p >= 0.5)), "f1": float(f1_score(y, p >= 0.5)),
            "roc_auc": float(roc_auc_score(y, p))}


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    df = generate_reviews()
    X_tr, X_te, y_tr, y_te, _, clean_te = train_test_split(
        df["text"].tolist(), df["label"].to_numpy(), df["clean_label"].to_numpy(),
        test_size=0.2, stratify=df["label"], random_state=0)
    report_ceiling = float((y_te == clean_te).mean())
    suites = challenge_sets()
    report = {"n_reviews": len(df), "examples": df.sample(5, random_state=1).to_dict(orient="records"),
              "label_noise_ceiling_accuracy": report_ceiling,
              "test": {}, "test_accuracy_vs_clean_labels": {}, "challenge_accuracy": {}}
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        report["test"][name] = evaluate(model, X_te, y_te)
        report["test_accuracy_vs_clean_labels"][name] = float(accuracy_score(clean_te, model.predict(X_te)))
        report["challenge_accuracy"][name] = {
            s: float(accuracy_score(d["label"], model.predict(d["text"].tolist()))) for s, d in suites.items()}

    best = fitted["scope_marking + word/char_lr"]
    report["top_features"] = top_features(best)
    pred = best.predict(X_te)
    wrong = np.flatnonzero(pred != y_te)
    report["errors_explained_by_label_noise"] = float((y_te[wrong] != clean_te[wrong]).mean())
    genuine = [i for i in wrong if y_te[i] == clean_te[i]][:8]
    report["genuine_error_examples"] = [{"text": X_te[i], "label": int(y_te[i])} for i in genuine]

    names = list(report["test"])
    cats = ["held-out test"] + list(suites)
    vals = np.array([[report["test"][n]["accuracy"]] + [report["challenge_accuracy"][n][s] for s in suites]
                     for n in names])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(cats))
    for j, n in enumerate(names):
        ax.bar(x + j * 0.2, vals[j], 0.2, label=n)
    ax.set_xticks(x + 0.3, cats); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    ax.set_title("Standard test accuracy vs. linguistic challenge sets")
    fig.tight_layout(); fig.savefig(RESULTS / "challenge_sets.png", dpi=120); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
