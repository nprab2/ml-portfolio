import numpy as np
import pandas as pd

from segmentation.data import generate_transactions
from segmentation.model import (CATEGORIES, MODEL_FEATURES, build_customer_features,
                                name_segments, to_model_matrix)


def test_features_are_complete_and_valid():
    tx, customers = generate_transactions(300, seed=0)
    f = build_customer_features(tx, customers, "2026-09-30")
    assert len(f) == 300 and not f[MODEL_FEATURES].isna().any().any()
    shares = f[[f"share_{c}" for c in CATEGORIES]].sum(axis=1)
    assert np.allclose(shares, 1.0)
    assert (f["recency_days"] >= 0).all()
    assert (f["recency_days"] <= f["tenure_days"]).all()


def test_shrinkage_keeps_single_order_customers_off_the_extremes():
    tx = pd.DataFrame({"customer_id": [0, 1, 1, 1], "category": ["home", "grocery", "fashion", "beauty"],
                       "amount": [10.0] * 4, "used_discount": [1, 0, 0, 0],
                       "timestamp": pd.to_datetime(["2026-09-01"] * 4)})
    cust = pd.DataFrame({"customer_id": [0, 1], "signup_date": pd.to_datetime(["2026-01-01"] * 2)})
    f = build_customer_features(tx, cust, "2026-09-30")
    assert 0 < f.loc[0, "discount_share"] < 1
    assert f.loc[0, "share_home"] < 1


def test_matrix_is_standardised_and_naming_is_total():
    tx, customers = generate_transactions(200, seed=1)
    X = to_model_matrix(build_customer_features(tx, customers, "2026-09-30"))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    prof = pd.DataFrame({"recency_days": [5, 300], "avg_order_value": [50, 60], "discount_share": [0.2, 0.2],
                         "share_grocery": [0.2, 0.2], "monetary_per_year": [100, 50]})
    assert set(name_segments(prof)) == {0, 1}
