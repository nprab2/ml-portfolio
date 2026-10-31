import numpy as np
from sklearn.model_selection import train_test_split

from housing.data import generate_houses
from housing.model import TARGET, HouseFeatures, SplitConformal, build_models


def test_generator_has_realistic_prices_and_missing_values():
    df = generate_houses(2000, seed=1)
    assert 200_000 < df[TARGET].median() < 1_500_000
    assert df["lot_sqft"].isna().any()


def test_engineered_features():
    df = generate_houses(10, seed=2)
    f = HouseFeatures().transform(df)
    assert np.allclose(f["dist_downtown"], np.hypot(df["x_km"] - 25, df["y_km"] - 25))
    assert (f["sqft_per_bed"] > 0).all()


def test_conformal_interval_coverage_is_close_to_nominal():
    df = generate_houses(4000, seed=3)
    X, y = df.drop(columns=TARGET), df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.3, random_state=0)
    conf = SplitConformal(build_models()["ridge_splines"], alpha=0.2).fit(X_fit, y_fit, X_cal, y_cal)
    lo, hi = conf.predict_interval(X_te)
    coverage = ((y_te >= lo) & (y_te <= hi)).mean()
    assert 0.74 < coverage < 0.88
    assert (lo < hi).all()
