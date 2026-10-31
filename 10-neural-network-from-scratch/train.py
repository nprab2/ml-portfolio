"""Train NumPy-only MLP and CNN on handwritten digits; compare with scikit-learn references.

Usage: python train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from nn.model import build_cnn, build_mlp, random_shift

RESULTS = Path(__file__).parent / "results"


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    X, y = load_digits(return_X_y=True)          # 1,797 real 8x8 handwritten digits (UCI)
    X = X / 16.0
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)
    img = lambda a: a.reshape(-1, 1, 8, 8)
    report = {"n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te), "models": {}}

    SEEDS = [0, 1, 2, 3, 4]   # 360 test images -> 1 image = 0.28%; average over seeds for honest comparisons

    def record(name, accs, seconds, n_params, **extra):
        report["models"][name] = {"test_accuracy_mean": float(np.mean(accs)), "test_accuracy_std": float(np.std(accs)),
                                  "train_seconds_per_run": round(float(np.mean(seconds)), 1), "parameters": n_params,
                                  **extra}

    t = time.time(); lr = LogisticRegression(max_iter=5000, C=1.0).fit(X_tr, y_tr)
    record("sklearn_logistic_regression", [(lr.predict(X_te) == y_te).mean()], [time.time() - t], lr.coef_.size + 10)
    accs, secs = [], []
    for sd in SEEDS:
        t = time.time()
        sk = MLPClassifier((256, 128), alpha=1e-4, max_iter=500, early_stopping=True, random_state=sd).fit(X_tr, y_tr)
        secs.append(time.time() - t); accs.append((sk.predict(X_te) == y_te).mean())
    record("sklearn_mlp (reference)", accs, secs, sum(w.size for w in sk.coefs_ + sk.intercepts_))

    histories, fitted = {}, {}
    for name, build, prep, aug in [
        ("numpy_mlp", build_mlp, lambda a: a, None),
        ("numpy_cnn", build_cnn, img, None),
        ("numpy_cnn + shift augmentation", build_cnn, img, random_shift),
    ]:
        accs, secs, epochs = [], [], []
        for sd in SEEDS:
            model = build(seed=sd)
            t = time.time()
            h = model.fit(prep(X_tr), y_tr, prep(X_va), y_va, epochs=80, patience=15, augment=aug, seed=sd)
            secs.append(time.time() - t); epochs.append(h["best_epoch"])
            accs.append((model.predict(prep(X_te)) == y_te).mean())
            if sd == 0:
                histories[name], fitted[name] = h, (model, prep)
        record(name, accs, secs, sum(p.size for p in model.params), mean_best_epoch=float(np.mean(epochs)))

    best_name = max((n for n in fitted), key=lambda n: report["models"][n]["test_accuracy_mean"])
    model, prep = fitted[best_name]
    pred = model.predict(prep(X_te))
    cm = confusion_matrix(y_te, pred)
    report["best_model"] = best_name
    report["confusion_matrix"] = cm.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for n, h in histories.items():
        axes[0].plot(h["val_loss"], label=f"{n} (val)")
        axes[0].plot(h["train_loss"], "--", alpha=0.5, color=axes[0].lines[-1].get_color())
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Learning curves (dashed = train)"); axes[0].legend(fontsize=7)
    axes[1].imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true"); axes[1].set_xticks(range(10)); axes[1].set_yticks(range(10))
    axes[1].set_title(f"Confusion matrix – {best_name}")
    wrong = np.flatnonzero(pred != y_te)[:8]
    axes[2].axis("off"); axes[2].set_title("Misclassified test digits (true → predicted)")
    for k, i in enumerate(wrong):
        ax = axes[2].inset_axes([(k % 4) * 0.25, 0.5 - (k // 4) * 0.5, 0.22, 0.42])
        ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{y_te[i]} → {pred[i]}", fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "training_report.png", dpi=120); plt.close(fig)

    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "confusion_matrix"}, indent=2))
