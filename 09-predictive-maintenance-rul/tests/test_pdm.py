import numpy as np
import pandas as pd

from pdm.data import simulate_fleet
from pdm.features import RegimeNormalizer, nasa_score, window_features


def test_rul_counts_down_to_zero_for_full_runs():
    df = simulate_fleet(5, seed=0)
    for _, g in df.groupby("engine_id"):
        assert g["rul"].iloc[-1] == 0
        assert (np.diff(g["rul"]) == -1).all()


def test_truncated_fleet_stops_before_failure():
    df = simulate_fleet(10, seed=1, truncate=True)
    assert (df.groupby("engine_id")["rul"].min() > 0).all()


def test_regime_normaliser_removes_regime_offsets():
    df = simulate_fleet(30, seed=2)
    out = RegimeNormalizer().fit(df).transform(df)
    means = out.groupby("regime")["s1"].mean()
    assert np.allclose(means, 0, atol=1e-6)


def test_window_features_are_causal():
    df = simulate_fleet(3, seed=3)
    sensors = ["s1", "s2"]
    full = window_features(df, sensors)
    cut = df.groupby("engine_id").head(50)
    part = window_features(cut, sensors)
    pd.testing.assert_frame_equal(full.loc[part.index].drop(columns="rul"), part.drop(columns="rul"))


def test_nasa_score_penalises_late_predictions_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40]) > 0
