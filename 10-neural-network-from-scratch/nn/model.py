"""Sequential model, Adam optimiser, training loop with early stopping, and augmentation."""
from __future__ import annotations

import copy

import numpy as np

from .layers import Conv2D, Dense, Dropout, Flatten, Layer, MaxPool2D, ReLU, softmax_cross_entropy


class Adam:
    def __init__(self, params, grads, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.grads = params, grads
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v):
            if self.wd and p.ndim > 1:            # decoupled weight decay on weights only (AdamW)
                p -= self.lr * self.wd * p
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            m_hat, v_hat = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Sequential:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    @property
    def params(self):
        return [p for l in self.layers for p in l.params]

    @property
    def grads(self):
        return [g for l in self.layers for g in l.grads]

    def forward(self, x, train=False):
        for l in self.layers:
            x = l.forward(x, train)
        return x

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)

    def predict_proba(self, x, batch=512):
        out = []
        for i in range(0, len(x), batch):
            z = self.forward(x[i:i + batch])
            z = np.exp(z - z.max(1, keepdims=True))
            out.append(z / z.sum(1, keepdims=True))
        return np.vstack(out)

    def predict(self, x):
        return self.predict_proba(x).argmax(1)

    def fit(self, X, y, X_val, y_val, epochs=60, batch_size=64, lr=2e-3, weight_decay=1e-4,
            patience=10, augment=None, seed=0, verbose=False) -> dict:
        rng = np.random.default_rng(seed)
        opt = Adam(self.params, self.grads, lr=lr, weight_decay=weight_decay)
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}
        best, best_state, wait = np.inf, None, 0
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            losses = []
            for i in range(0, len(X), batch_size):
                b = idx[i:i + batch_size]
                xb = augment(X[b], rng) if augment else X[b]
                loss, grad = softmax_cross_entropy(self.forward(xb, train=True), y[b])
                self.backward(grad)
                opt.step()
                losses.append(loss)
            # cosine learning-rate decay
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * (ep + 1) / epochs))
            val_loss, _ = softmax_cross_entropy(self.forward(X_val), y_val)
            hist["train_loss"].append(float(np.mean(losses)))
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(float((self.predict(X_val) == y_val).mean()))
            if verbose:
                print(f"epoch {ep:3d}  train {hist['train_loss'][-1]:.4f}  val {val_loss:.4f}  acc {hist['val_acc'][-1]:.4f}")
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                best_state = copy.deepcopy([p.copy() for p in self.params])
            else:
                wait += 1
                if wait >= patience:
                    break
        for p, saved in zip(self.params, best_state):      # restore best weights
            p[...] = saved
        hist["best_epoch"] = int(np.argmin(hist["val_loss"]))
        return hist


def build_mlp(n_in=64, hidden=(256, 128), n_out=10, dropout=0.2, seed=0) -> Sequential:
    rng = np.random.default_rng(seed)
    layers, prev = [], n_in
    for h in hidden:
        layers += [Dense(prev, h, rng), ReLU(), Dropout(dropout, rng)]
        prev = h
    layers.append(Dense(prev, n_out, rng))
    return Sequential(layers)


def build_cnn(n_out=10, seed=0) -> Sequential:
    """Input (N, 1, 8, 8) -> conv16 -> pool -> conv32 -> pool -> dense 64 -> 10."""
    rng = np.random.default_rng(seed)
    return Sequential([
        Conv2D(1, 16, 3, rng), ReLU(), MaxPool2D(),          # 16 x 4 x 4
        Conv2D(16, 32, 3, rng), ReLU(), MaxPool2D(),         # 32 x 2 x 2
        Flatten(), Dense(128, 64, rng), ReLU(), Dropout(0.3, rng), Dense(64, n_out, rng),
    ])


def random_shift(x: np.ndarray, rng: np.random.Generator, max_shift: int = 1) -> np.ndarray:
    """Data augmentation: translate each image by up to ``max_shift`` pixels (zero fill)."""
    imgs = x.reshape(len(x), 8, 8) if x.ndim == 2 else x[:, 0]
    out = np.zeros_like(imgs)
    for k, (dy, dx) in enumerate(rng.integers(-max_shift, max_shift + 1, (len(x), 2))):
        src = imgs[k, max(0, -dy):8 - max(0, dy), max(0, -dx):8 - max(0, dx)]
        out[k, max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out.reshape(x.shape)
