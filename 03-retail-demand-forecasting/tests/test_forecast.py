import numpy as np
import pandas as pd

from forecast.data import generate_sales
from forecast.features import HORIZON, bias, coverage, make_features, wape


def test_features_never_use_data_inside_the_horizon():
    df = generate_sales(n_stores=1, n_items=1, periods=200, seed=0)
    f = make_features(df)
    # Perturb the last HORIZON days of sales: features on those days must not change.
    tampered = df.copy()
    tampered.loc[tampered.index[-HORIZON:], "sales"] += 10_000
    g = make_features(tampered)
    cols = [c for c in f.columns if c.startswith(("lag_", "roll_", "dow_mean"))]
    pd.testing.assert_frame_equal(f[cols].tail(HORIZON), g[cols].tail(HORIZON))


def test_metrics():
    y, yhat = np.array([10, 0, 10]), np.array([12, 1, 8])
    assert wape(y, yhat) == 5 / 20
    assert bias(y, yhat) == 1 / 20
    assert coverage(y, yhat - 3, yhat + 3) == 1.0


def test_generator_shape_and_counts():
    df = generate_sales(n_stores=2, n_items=3, periods=30)
    assert len(df) == 2 * 3 * 30
    assert (df["sales"] >= 0).all() and df["sales"].dtype.kind == "i"
