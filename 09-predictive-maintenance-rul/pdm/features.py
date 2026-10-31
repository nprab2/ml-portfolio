"""Operating-regime normalisation, sensor selection and windowed features for RUL models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RUL_CAP = 125  # piecewise-linear target: early-life degradation is not observable


class RegimeNormalizer:
    """Cluster operating settings into regimes, then z-score each sensor *within* its regime.

    Without this step the regime shift (≈6σ) swamps the degradation signal (≈1–3σ).
    """

    def __init__(self, n_regimes: int = 3, seed: int = 0):
        self.n_regimes, self.seed = n_regimes, seed

    def fit(self, df: pd.DataFrame):
        self.settings_ = [c for c in df.columns if c.startswith("setting")]
        self.sensors_ = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
        self.km_ = KMeans(self.n_regimes, n_init=10, random_state=self.seed).fit(df[self.settings_])
        reg = self.km_.labels_
        self.mean_ = df[self.sensors_].groupby(reg).mean()
        self.std_ = df[self.sensors_].groupby(reg).std().replace(0, 1)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        reg = self.km_.predict(df[self.settings_])
        out["regime"] = reg
        out[self.sensors_] = (df[self.sensors_].to_numpy() - self.mean_.loc[reg].to_numpy()) / self.std_.loc[reg].to_numpy()
        return out


def prognostic_score(df: pd.DataFrame, sensors: list[str]) -> pd.Series:
    """|Spearman correlation| between each sensor and RUL in late life – a cheap prognosability measure."""
    late = df[df["rul"] < 150]
    return late[sensors].corrwith(late["rul"], method="spearman").abs().sort_values(ascending=False)


def window_features(df: pd.DataFrame, sensors: list[str], windows=(10, 30)) -> pd.DataFrame:
    """Causal rolling features per engine (only past and current cycles are used)."""
    df = df.sort_values(["engine_id", "cycle"])
    g = df.groupby("engine_id", sort=False)
    feats = {"cycle": df["cycle"].to_numpy()}
    for s in sensors:
        for w in windows:
            roll = g[s].rolling(w, min_periods=1)
            feats[f"{s}_mean{w}"] = roll.mean().to_numpy()
        feats[f"{s}_std30"] = g[s].rolling(30, min_periods=2).std().to_numpy()
        feats[f"{s}_ewm"] = g[s].transform(lambda x: x.ewm(alpha=0.1).mean()).to_numpy()
        # Slope over the last 30 cycles: cov(x, t) / var(t)
        t = df["cycle"].astype(float)
        cov = (df[s] * t).groupby(df["engine_id"]).rolling(30, min_periods=5).mean().to_numpy() - \
              g[s].rolling(30, min_periods=5).mean().to_numpy() * t.groupby(df["engine_id"]).rolling(30, min_periods=5).mean().to_numpy()
        var = (t * t).groupby(df["engine_id"]).rolling(30, min_periods=5).mean().to_numpy() - \
              t.groupby(df["engine_id"]).rolling(30, min_periods=5).mean().to_numpy() ** 2
        feats[f"{s}_slope30"] = cov / np.where(var > 0, var, np.nan)
    out = pd.DataFrame(feats, index=df.index)
    out["engine_id"] = df["engine_id"].to_numpy()
    if "rul" in df:
        out["rul"] = df["rul"].to_numpy()
    return out


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score: late predictions (overestimating RUL) are penalised more."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
