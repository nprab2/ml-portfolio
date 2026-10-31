"""Recommender models implemented from scratch with NumPy/SciPy, plus ranking metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class BiasBaseline:
    """r_ui ≈ μ + b_u + b_i, fitted by alternating regularised least squares."""

    def __init__(self, reg: float = 10.0, n_iter: int = 10):
        self.reg, self.n_iter = reg, n_iter

    def fit(self, df: pd.DataFrame, n_users: int, n_items: int):
        u, i, r = df["user_id"].to_numpy(), df["item_id"].to_numpy(), df["rating"].to_numpy(float)
        self.mu = r.mean()
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        cu, ci = np.bincount(u, minlength=n_users), np.bincount(i, minlength=n_items)
        for _ in range(self.n_iter):
            self.bi = np.bincount(i, r - self.mu - self.bu[u], minlength=n_items) / (ci + self.reg)
            self.bu = np.bincount(u, r - self.mu - self.bi[i], minlength=n_users) / (cu + self.reg)
        return self

    def predict(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i], 1, 5)


class BiasedALS(BiasBaseline):
    """Matrix factorisation of the bias-model residuals with alternating least squares:
    r_ui ≈ μ + b_u + b_i + p_u·q_i.  Each ALS step is an exact ridge-regression solve."""

    def __init__(self, n_factors: int = 32, reg: float = 10.0, factor_reg: float = 8.0,
                 n_iter: int = 12, seed: int = 0):
        super().__init__(reg)
        self.k, self.lam, self.als_iter, self.seed = n_factors, factor_reg, n_iter, seed

    def fit(self, df, n_users, n_items):
        super().fit(df, n_users, n_items)
        u, i = df["user_id"].to_numpy(), df["item_id"].to_numpy()
        resid = df["rating"].to_numpy(float) - (self.mu + self.bu[u] + self.bi[i])
        R = sparse.csr_matrix((resid, (u, i)), shape=(n_users, n_items))
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        self.history = []
        for _ in range(self.als_iter):
            self.P = self._solve(R, self.Q)
            self.Q = self._solve(Rt, self.P)
            pred = np.einsum("ij,ij->i", self.P[u], self.Q[i])
            self.history.append(float(np.sqrt(np.mean((resid - pred) ** 2))))
        return self

    def _solve(self, R: sparse.csr_matrix, Y: np.ndarray) -> np.ndarray:
        X = np.zeros((R.shape[0], self.k))
        eye = self.lam * np.eye(self.k)
        for row in range(R.shape[0]):
            s, e = R.indptr[row], R.indptr[row + 1]
            if s == e:
                continue
            Yi = Y[R.indices[s:e]]
            X[row] = np.linalg.solve(Yi.T @ Yi + eye * (e - s), Yi.T @ R.data[s:e])  # weighted-λ reg
        return X

    def predict(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i]), 1, 5)

    def score_all(self, users: np.ndarray) -> np.ndarray:
        return self.bi[None, :] + self.P[users] @ self.Q.T   # user bias & μ don't change ranking


class ImplicitALS:
    """Implicit-feedback ALS (Hu, Koren & Volinsky, 2008).

    Models *whether* a user engages with an item, with confidence c_ui = 1 + α·r_ui.
    Uses the YᵀY + Yᵤᵀ(Cᵤ − I)Yᵤ trick so each solve touches only observed items.
    Because it also learns from what users did NOT watch, it ranks far better than
    a rating-prediction model when exposure is missing-not-at-random.
    """

    def __init__(self, n_factors: int = 64, reg: float = 5.0, alpha: float = 10.0,
                 n_iter: int = 12, seed: int = 0):
        self.k, self.reg, self.alpha, self.n_iter, self.seed = n_factors, reg, alpha, n_iter, seed

    def fit(self, df, n_users, n_items):
        conf = 1 + self.alpha * df["rating"].to_numpy(float) / 5
        C = sparse.csr_matrix((conf, (df["user_id"], df["item_id"])), shape=(n_users, n_items))
        Ct = C.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.01, (n_users, self.k))
        self.Q = rng.normal(0, 0.01, (n_items, self.k))
        for _ in range(self.n_iter):
            self.P = self._solve(C, self.Q)
            self.Q = self._solve(Ct, self.P)
        return self

    def _solve(self, C, Y):
        YtY = Y.T @ Y + self.reg * np.eye(self.k)
        X = np.zeros((C.shape[0], self.k))
        for row in range(C.shape[0]):
            s, e = C.indptr[row], C.indptr[row + 1]
            if s == e:
                continue
            Yi, c = Y[C.indices[s:e]], C.data[s:e]
            A = YtY + (Yi.T * (c - 1)) @ Yi
            X[row] = np.linalg.solve(A, Yi.T @ c)       # preference p_ui = 1 for observed items
        return X

    def score_all(self, users):
        return self.P[users] @ self.Q.T


class ItemKNN:
    """Item-based collaborative filtering with shrunk, mean-centred cosine similarity."""

    def __init__(self, k: int = 50, shrink: float = 20.0):
        self.k, self.shrink = k, shrink

    def fit(self, df, n_users, n_items):
        u, i, r = df["user_id"].to_numpy(), df["item_id"].to_numpy(), df["rating"].to_numpy(float)
        user_mean = np.bincount(u, r, n_users) / np.maximum(np.bincount(u, minlength=n_users), 1)
        X = sparse.csr_matrix((r - user_mean[u], (u, i)), shape=(n_users, n_items))
        B = sparse.csr_matrix((np.ones_like(r), (u, i)), shape=(n_users, n_items))
        num = (X.T @ X).toarray()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0))).ravel() + 1e-9
        co = (B.T @ B).toarray()
        S = num / np.outer(norms, norms) * (co / (co + self.shrink))
        np.fill_diagonal(S, 0)
        # keep top-k neighbours per item
        drop = np.argpartition(-S, self.k, axis=1)[:, self.k:]
        np.put_along_axis(S, drop, 0, axis=1)
        self.S, self.X, self.B = S, X, B
        return self

    def score_all(self, users):
        return np.asarray(self.X[users] @ self.S.T)


class Popularity:
    def fit(self, df, n_users, n_items):
        self.pop = np.bincount(df["item_id"], minlength=n_items).astype(float)
        return self

    def score_all(self, users):
        return np.tile(self.pop, (len(users), 1))


def ranking_metrics(model, train: pd.DataFrame, test: pd.DataFrame, n_items: int, k: int = 10,
                    relevant_threshold: int = 4) -> dict:
    """Precision@k, Recall@k, NDCG@k, MAP@k and catalogue coverage over all items,
    excluding items a user already rated in training."""
    rel = test[test["rating"] >= relevant_threshold].groupby("user_id")["item_id"].apply(set)
    users = rel.index.to_numpy()
    scores = model.score_all(users).astype(float)
    seen = train[train["user_id"].isin(users)]
    row = pd.Series(np.arange(len(users)), index=users)
    scores[row[seen["user_id"]].to_numpy(), seen["item_id"].to_numpy()] = -np.inf
    top = np.argsort(-scores, axis=1)[:, :k]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    p, r, ndcg, ap = [], [], [], []
    for idx, u in enumerate(users):
        hits = np.isin(top[idx], list(rel[u]))
        n_rel = len(rel[u])
        p.append(hits.mean()); r.append(hits.sum() / n_rel)
        ndcg.append((hits * discounts).sum() / discounts[: min(n_rel, k)].sum())
        prec_at = np.cumsum(hits) / np.arange(1, k + 1)
        ap.append((prec_at * hits).sum() / min(n_rel, k))
    return {f"precision@{k}": float(np.mean(p)), f"recall@{k}": float(np.mean(r)),
            f"ndcg@{k}": float(np.mean(ndcg)), f"map@{k}": float(np.mean(ap)),
            "catalog_coverage": float(len(np.unique(top)) / n_items)}
