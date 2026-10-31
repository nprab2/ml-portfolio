"""A/B testing toolkit: power analysis, frequentist & Bayesian tests, CUPED,
sequential testing (mSPRT) and sample-ratio-mismatch checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- design
def sample_size_proportions(p_base: float, mde_rel: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users *per arm* to detect a relative lift ``mde_rel`` on a conversion rate (two-sided)."""
    p1, p2 = p_base, p_base * (1 + mde_rel)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(np.ceil(n))


def srm_check(n_a: int, n_b: int, expected_ratio: float = 0.5, threshold: float = 0.001) -> dict:
    """Sample-ratio mismatch: a tiny p-value means the randomisation or logging is broken."""
    total = n_a + n_b
    chi2, p = stats.chisquare([n_a, n_b], [total * expected_ratio, total * (1 - expected_ratio)])
    return {"chi2": float(chi2), "p_value": float(p), "srm_detected": bool(p < threshold)}


# --------------------------------------------------------------------------- frequentist
@dataclass
class TestResult:
    estimate: float          # absolute difference B - A
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float

    def as_dict(self):
        return {k: float(v) for k, v in self.__dict__.items()}


def ztest_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05) -> TestResult:
    pa, pb = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    p_value = 2 * stats.norm.sf(abs(pb - pa) / se_pooled)
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = stats.norm.ppf(1 - alpha / 2)
    return TestResult(pb - pa, (pb - pa) / pa, pb - pa - z * se, pb - pa + z * se, p_value)


def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    t = stats.ttest_ind(b, a, equal_var=False)
    ci = t.confidence_interval(1 - alpha)
    diff = b.mean() - a.mean()
    return TestResult(diff, diff / a.mean(), ci.low, ci.high, t.pvalue)


def cuped(y: np.ndarray, x_pre: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED (Deng et al., 2013): y' = y − θ(x − mean(x)), θ = cov(y,x)/var(x).

    Uses pre-experiment data, so it cannot bias the treatment effect, but removes
    variance explained by pre-existing user differences."""
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean()), float(theta)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, stat=np.median, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any statistic difference (e.g., median revenue)."""
    rng = np.random.default_rng(seed)
    diffs = np.array([stat(rng.choice(b, b.size)) - stat(rng.choice(a, a.size)) for _ in range(n_boot)])
    return float(stat(b) - stat(a)), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


# --------------------------------------------------------------------------- Bayesian
def bayesian_proportions(conv_a, n_a, conv_b, n_b, prior=(1, 1), draws: int = 200_000, seed: int = 0) -> dict:
    """Beta-binomial model. Reports P(B > A), expected loss of choosing each arm, and a credible interval."""
    rng = np.random.default_rng(seed)
    a = rng.beta(prior[0] + conv_a, prior[1] + n_a - conv_a, draws)
    b = rng.beta(prior[0] + conv_b, prior[1] + n_b - conv_b, draws)
    lift = b / a - 1
    return {"prob_b_better": float((b > a).mean()),
            "expected_loss_choose_b": float(np.maximum(a - b, 0).mean()),
            "expected_loss_choose_a": float(np.maximum(b - a, 0).mean()),
            "lift_credible_interval_95": [float(np.quantile(lift, 0.025)), float(np.quantile(lift, 0.975))]}


# --------------------------------------------------------------------------- sequential
def msprt_statistic(xa: np.ndarray, xb: np.ndarray, tau2: float, sigma2: float) -> np.ndarray:
    """Mixture SPRT likelihood ratio after each pair of observations (Johari et al., 2017).

    Under H0 (no difference) the statistic is a martingale, so rejecting the first
    time it exceeds 1/α keeps the false-positive rate ≤ α *no matter how often you peek*.
    """
    n = np.arange(1, len(xa) + 1)
    diff = np.cumsum(xb - xa) / n
    V = 2 * sigma2
    return np.sqrt(V / (V + n * tau2)) * np.exp(n ** 2 * tau2 * diff ** 2 / (2 * V * (V + n * tau2)))


def peeking_simulation(n_sims: int = 2000, n_per_arm: int = 5000, n_looks: int = 50, p: float = 0.1,
                       lift: float = 0.0, alpha: float = 0.05, seed: int = 0) -> dict:
    """Compare stopping rules when analysts check results repeatedly.

    Returns rejection rates for: a single fixed-horizon test, naive peeking with a
    z-test at every look, and the always-valid mSPRT.
    """
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    sigma2 = p * (1 - p)
    tau2 = (0.1 * p) ** 2           # prior scale on the effect: ~10% relative lift
    fixed = naive = seq = 0
    stop_n = []
    for _ in range(n_sims):
        xa = (rng.random(n_per_arm) < p).astype(float)
        xb = (rng.random(n_per_arm) < p * (1 + lift)).astype(float)
        ca, cb = np.cumsum(xa), np.cumsum(xb)
        za = []
        for m in looks:
            pa, pb = ca[m - 1] / m, cb[m - 1] / m
            pooled = (pa + pb) / 2
            se = np.sqrt(2 * pooled * (1 - pooled) / m) or 1e-12
            za.append(abs(pb - pa) / se)
        za = np.array(za)
        fixed += za[-1] > z_crit
        naive += (za > z_crit).any()
        lam = msprt_statistic(xa, xb, tau2, sigma2)[looks - 1]
        hit = np.flatnonzero(lam >= 1 / alpha)
        if hit.size:
            seq += 1
            stop_n.append(looks[hit[0]])
    return {"fixed_horizon": fixed / n_sims, "naive_peeking": naive / n_sims, "msprt": seq / n_sims,
            "msprt_median_stop_n": float(np.median(stop_n)) if stop_n else None}
