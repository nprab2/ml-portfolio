"""Neural-network layers with hand-written forward and backward passes (NumPy only)."""
from __future__ import annotations

import numpy as np


class Layer:
    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))     # He initialisation
        self.b = np.zeros(n_out)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.grads[0][...] = self.x.T @ grad
        self.grads[1][...] = grad.sum(0)
        return grad @ self.W.T


class ReLU(Layer):
    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Dropout(Layer):
    """Inverted dropout: scale at train time so inference needs no change."""

    def __init__(self, p: float, rng: np.random.Generator):
        self.p, self.rng = p, rng

    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


def _im2col(x: np.ndarray, k: int, pad: int) -> np.ndarray:
    """(N, C, H, W) -> (N*H_out*W_out, C*k*k) patches for stride-1 convolution."""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    s = xp.strides
    patches = np.lib.stride_tricks.as_strided(xp, (n, c, ho, wo, k, k), (s[0], s[1], s[2], s[3], s[2], s[3]))
    return patches.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)


class Conv2D(Layer):
    """Stride-1 'same' convolution implemented as a matrix multiply via im2col."""

    def __init__(self, c_in: int, c_out: int, k: int, rng: np.random.Generator):
        self.k, self.pad = k, k // 2
        self.W = rng.normal(0, np.sqrt(2 / (c_in * k * k)), (c_out, c_in, k, k))
        self.b = np.zeros(c_out)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x, train=True):
        self.shape = x.shape
        n, _, h, w = x.shape
        self.cols = _im2col(x, self.k, self.pad)
        out = self.cols @ self.W.reshape(len(self.b), -1).T + self.b
        return out.reshape(n, h, w, -1).transpose(0, 3, 1, 2)

    def backward(self, grad):
        n, c, h, w = self.shape
        g = grad.transpose(0, 2, 3, 1).reshape(-1, len(self.b))
        self.grads[0][...] = (g.T @ self.cols).reshape(self.W.shape)
        self.grads[1][...] = g.sum(0)
        dcols = (g @ self.W.reshape(len(self.b), -1)).reshape(n, h, w, c, self.k, self.k)
        dx = np.zeros((n, c, h + 2 * self.pad, w + 2 * self.pad))
        for i in range(self.k):                        # col2im: scatter-add each kernel offset
            for j in range(self.k):
                dx[:, :, i:i + h, j:j + w] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx[:, :, self.pad:self.pad + h, self.pad:self.pad + w]


class MaxPool2D(Layer):
    """2x2 max pooling with stride 2."""

    def forward(self, x, train=True):
        n, c, h, w = x.shape
        xr = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = xr.max(axis=(3, 5))
        self.mask = xr == out[:, :, :, None, :, None]
        self.mask = self.mask / self.mask.sum(axis=(3, 5), keepdims=True)   # split ties evenly
        self.shape = x.shape
        return out

    def backward(self, grad):
        return (self.mask * grad[:, :, :, None, :, None]).reshape(self.shape)


class Flatten(Layer):
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Numerically stable mean cross-entropy and its gradient w.r.t. the logits."""
    z = logits - logits.max(1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -log_p[np.arange(len(y)), y].mean()
    grad = np.exp(log_p)
    grad[np.arange(len(y)), y] -= 1
    return float(loss), grad / len(y)
