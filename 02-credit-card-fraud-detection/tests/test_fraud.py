import numpy as np
import pandas as pd

from fraud.data import add_behavioural_features, generate_transactions
from fraud.model import dollars_saved, precision_at_k, recall_at_fpr


def test_generator_is_imbalanced_and_sorted():
    df = generate_transactions(200, days=10, seed=1)
    assert 0 < df["is_fraud"].mean() < 0.05
    assert df["timestamp"].is_monotonic_increasing


def test_behavioural_features_do_not_look_ahead():
    df = pd.DataFrame({"customer_id": [1, 1, 1], "timestamp": [0.0, 100.0, 7200.0],
                       "amount": [10.0, 30.0, 1000.0], "merchant_category": ["gas"] * 3,
                       "distance_from_home_km": [1.0] * 3, "card_present": [1] * 3,
                       "new_device": [0] * 3, "is_fraud": [0, 0, 1]})
    f = add_behavioural_features(df)
    assert np.isnan(f.loc[0, "amount_vs_customer_mean"])          # no history yet
    assert f.loc[1, "amount_vs_customer_mean"] == 3.0              # 30 / mean(10)
    assert f.loc[2, "amount_vs_customer_mean"] == 1000 / 20         # uses only past rows
    assert list(f["tx_last_1h"]) == [0, 1, 0]                      # window excludes current tx


def test_metrics():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    assert precision_at_k(y, s, 2) == 1.0
    assert recall_at_fpr(y, s, 0.01) == 1.0
    assert dollars_saved(y, s, np.array([1, 1, 1, 100, 50.0]), review_budget=2) == 140.0
