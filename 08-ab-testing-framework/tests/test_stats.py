import numpy as np
from scipy import stats

from abtest.stats import (bayesian_proportions, cuped, msprt_statistic, sample_size_proportions,
                          srm_check, welch_ttest, ztest_proportions)


def test_sample_size_matches_textbook_value():
    # 10% -> 12% conversion, alpha 0.05, power 0.8: standard calculators give ~3,841 per arm
    assert abs(sample_size_proportions(0.10, 0.20) - 3841) < 15


def test_ztest_agrees_with_chi_square():
    res = ztest_proportions(200, 2000, 250, 2000)
    chi = stats.chi2_contingency([[200, 1800], [250, 1750]], correction=False)
    assert np.isclose(res.p_value, chi.pvalue)
    assert res.ci_low < res.estimate < res.ci_high


def test_srm_flags_broken_split():
    assert srm_check(50_000, 50_300)["srm_detected"] is False
    assert srm_check(50_000, 52_000)["srm_detected"] is True


def test_cuped_preserves_effect_and_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 3, 20_000)
    treat = rng.random(20_000) < 0.5
    y = x + rng.normal(0, 1, 20_000) + 0.5 * treat
    adj, theta = cuped(y, x)
    assert 0.9 < theta < 1.1
    assert np.var(adj) < 0.2 * np.var(y)
    assert abs(welch_ttest(adj[~treat], adj[treat]).estimate - 0.5) < 0.05


def test_bayesian_is_symmetric_with_identical_data():
    r = bayesian_proportions(100, 1000, 100, 1000)
    assert abs(r["prob_b_better"] - 0.5) < 0.01


def test_msprt_controls_type_one_error():
    rng = np.random.default_rng(1)
    rejections = 0
    for _ in range(300):
        a, b = rng.random(3000) < 0.1, rng.random(3000) < 0.1
        lam = msprt_statistic(a.astype(float), b.astype(float), tau2=1e-4, sigma2=0.09)
        rejections += (lam >= 20).any()          # checked after *every* observation
    assert rejections / 300 <= 0.05
