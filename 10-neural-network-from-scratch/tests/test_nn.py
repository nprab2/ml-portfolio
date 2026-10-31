"""Numerical gradient checks: the strongest evidence that hand-written backprop is correct."""
import numpy as np

from nn.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, softmax_cross_entropy
from nn.model import Sequential, build_mlp, random_shift


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check_model(model, x, y):
    loss, grad = softmax_cross_entropy(model.forward(x, train=True), y)
    model.backward(grad)
    f = lambda: softmax_cross_entropy(model.forward(x, train=True), y)[0]
    for p, g in zip(model.params, model.grads):
        assert np.allclose(g, numeric_grad(f, p), atol=1e-5, rtol=1e-4)


def test_dense_relu_gradients():
    rng = np.random.default_rng(0)
    model = Sequential([Dense(5, 7, rng), ReLU(), Dense(7, 3, rng)])
    check_model(model, rng.normal(size=(4, 5)), np.array([0, 2, 1, 2]))


def test_conv_pool_gradients():
    rng = np.random.default_rng(1)
    model = Sequential([Conv2D(1, 2, 3, rng), ReLU(), MaxPool2D(), Flatten(), Dense(8, 3, rng)])
    check_model(model, rng.normal(size=(2, 1, 4, 4)), np.array([1, 0]))


def test_conv_input_gradient():
    rng = np.random.default_rng(2)
    conv = Conv2D(2, 3, 3, rng)
    x = rng.normal(size=(1, 2, 4, 4))
    w = rng.normal(size=(1, 3, 4, 4))
    conv.forward(x)
    dx = conv.backward(w)
    assert np.allclose(dx, numeric_grad(lambda: float((conv.forward(x) * w).sum()), x), atol=1e-6)


def test_softmax_cross_entropy_matches_definition():
    logits = np.array([[2.0, 1.0, 0.1]])
    p = np.exp(logits) / np.exp(logits).sum()
    loss, _ = softmax_cross_entropy(logits, np.array([0]))
    assert np.isclose(loss, -np.log(p[0, 0]))


def test_mlp_can_overfit_tiny_dataset():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(30, 64)), rng.integers(0, 10, 30)
    model = build_mlp(dropout=0.0)
    model.fit(X, y, X, y, epochs=200, lr=5e-3, weight_decay=0, patience=200)
    assert (model.predict(X) == y).mean() == 1.0


def test_random_shift_preserves_pixel_mass_when_no_clipping():
    x = np.zeros((1, 64)); x[0, 27] = 1.0             # a single pixel in the middle
    out = random_shift(x, np.random.default_rng(0))
    assert out.sum() == 1.0 and out.shape == x.shape
