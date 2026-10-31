"""Full experiment read-out plus simulation studies that validate the methods.

Usage: python run_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abtest.simulate import simulate_experiment
from abtest.stats import (bayesian_proportions, bootstrap_ci, cuped, peeking_simulation,
                          sample_size_proportions, srm_check, welch_ttest, ztest_proportions)

RESULTS = Path(__file__).parent / "results"


def experiment_readout() -> dict:
    df = simulate_experiment()
    A, B = df[df.group == "A"], df[df.group == "B"]
    out = {"design": {"users_per_arm_needed_for_8pct_mde": sample_size_proportions(0.04, 0.08)},
           "srm": srm_check(len(A), len(B)),
           "n": {"A": len(A), "B": len(B)},
           "conversion": {"A": float(A.converted.mean()), "B": float(B.converted.mean()),
                          "ztest": ztest_proportions(A.converted.sum(), len(A), B.converted.sum(), len(B)).as_dict(),
                          "bayesian": bayesian_proportions(A.converted.sum(), len(A), B.converted.sum(), len(B))}}
    raw = welch_ttest(A.revenue.to_numpy(), B.revenue.to_numpy())
    adj, theta = cuped(df.revenue.to_numpy(), df.pre_revenue.to_numpy())
    adj_res = welch_ttest(adj[df.group == "A"], adj[df.group == "B"])
    out["revenue_per_user"] = {
        "raw_welch": raw.as_dict(), "cuped_welch": adj_res.as_dict(), "cuped_theta": theta,
        "variance_reduction": float(1 - np.var(adj) / np.var(df.revenue)),
        "ci_width_raw": raw.ci_high - raw.ci_low, "ci_width_cuped": adj_res.ci_high - adj_res.ci_low,
        "median_diff_bootstrap": bootstrap_ci(A.revenue.to_numpy(), B.revenue.to_numpy())}
    return out


def aa_validation(n_tests: int = 1000, seed: int = 1) -> dict:
    """Run many A/A tests – a correct test rejects ~5% of the time and has uniform p-values."""
    rng = np.random.default_rng(seed)
    pvals = []
    for _ in range(n_tests):
        a, b = rng.random(10_000) < 0.05, rng.random(10_000) < 0.05
        pvals.append(ztest_proportions(a.sum(), a.size, b.sum(), b.size).p_value)
    pvals = np.array(pvals)
    return {"false_positive_rate": float((pvals < 0.05).mean()), "pvalues": pvals}


def power_study(seed: int = 2) -> dict:
    """Empirical power of the z-test vs. the analytic sample-size formula."""
    rng = np.random.default_rng(seed)
    n_needed = sample_size_proportions(0.05, 0.10)
    sizes = (np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0]) * n_needed).astype(int)
    emp = []
    for n in sizes:
        hits = 0
        for _ in range(500):
            a = rng.binomial(n, 0.05); b = rng.binomial(n, 0.055)
            hits += ztest_proportions(a, n, b, n).p_value < 0.05
        emp.append(hits / 500)
    return {"analytic_n_for_80pct_power": n_needed, "sizes": sizes.tolist(), "empirical_power": emp}


def main() -> dict:
    RESULTS.mkdir(exist_ok=True)
    report = {"experiment": experiment_readout()}
    aa = aa_validation()
    report["aa_false_positive_rate"] = aa["false_positive_rate"]
    power = power_study()
    report["power_study"] = power
    report["peeking_null"] = peeking_simulation(lift=0.0)
    report["peeking_10pct_lift"] = peeking_simulation(lift=0.10, n_per_arm=30_000, n_sims=500, seed=3)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(aa["pvalues"], bins=20, edgecolor="w")
    axes[0].set_title(f"A/A p-values (FPR = {aa['false_positive_rate']:.1%})"); axes[0].set_xlabel("p-value")
    axes[1].plot(power["sizes"], power["empirical_power"], "o-", label="simulated")
    axes[1].axhline(0.8, color="grey", ls=":"); axes[1].axvline(power["analytic_n_for_80pct_power"], color="r", ls="--",
                                                                 label="formula n for 80%")
    axes[1].set_xlabel("users per arm"); axes[1].set_ylabel("power"); axes[1].set_title("Power: 5% → 5.5% conversion")
    axes[1].legend()
    pn = report["peeking_null"]
    axes[2].bar(["fixed horizon", "naive peeking\n(50 looks)", "mSPRT\n(50 looks)"],
                [pn["fixed_horizon"], pn["naive_peeking"], pn["msprt"]], color=["C0", "C3", "C2"])
    axes[2].axhline(0.05, color="k", ls="--", lw=0.8); axes[2].set_ylabel("false positive rate")
    axes[2].set_title("The cost of peeking (no true effect)")
    fig.tight_layout(); fig.savefig(RESULTS / "validation.png", dpi=120); plt.close(fig)

    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
