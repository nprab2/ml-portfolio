import numpy as np
from sklearn.metrics import roc_auc_score

from churn.data import generate_churn_data
from churn.model import TARGET, build_gbm, expected_profit, optimal_threshold


def test_data_is_reproducible_and_well_formed():
    a, b = generate_churn_data(500, seed=1), generate_churn_data(500, seed=1)
    assert a.equals(b)
    assert set(a[TARGET].unique()) <= {0, 1}
    assert a["tenure_months"].between(1, 72).all()


def test_model_beats_random_and_handles_missing_values():
    df = generate_churn_data(3000, seed=0)
    X, y = df.drop(columns=[TARGET, "customer_id"]), df[TARGET].values
    assert X["total_charges"].isna().any()
    model = build_gbm(calibrate=False).fit(X[:2000], y[:2000])
    proba = model.predict_proba(X[2000:])[:, 1]
    assert roc_auc_score(y[2000:], proba) > 0.75


def test_profit_function_matches_hand_calculation():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    # Threshold 0.5 targets 2 customers, 1 true churner: 1*0.35*600 - 2*50 = 110
    assert expected_profit(y, p, 0.5) == 110.0
    thr, profit = optimal_threshold(y, p)
    assert profit >= 110.0 and 0 < thr < 1
