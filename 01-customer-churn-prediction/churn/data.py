"""Synthetic telecom customer data generator.

The generator encodes realistic, documented churn drivers (short tenure,
month-to-month contracts, fiber service complaints, electronic-check payments,
many support calls) so that model behaviour can be checked against known truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
INTERNET = ["dsl", "fiber", "none"]
PAYMENTS = ["electronic-check", "mailed-check", "bank-transfer", "credit-card"]


def generate_churn_data(n_customers: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of customers with a binary ``churned`` label."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n_customers, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n_customers, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENTS, n_customers, p=[0.35, 0.20, 0.22, 0.23])

    # Longer contracts correlate with longer tenure.
    tenure_scale = np.select(
        [contract == "month-to-month", contract == "one-year"], [14, 30], default=48
    )
    tenure = np.clip(rng.exponential(tenure_scale), 1, 72).round().astype(int)

    base_charge = np.select(
        [internet == "fiber", internet == "dsl"], [80.0, 55.0], default=22.0
    )
    num_products = rng.integers(1, 6, n_customers)
    monthly_charges = np.round(base_charge + num_products * 4 + rng.normal(0, 6, n_customers), 2)
    monthly_charges = np.clip(monthly_charges, 18, 130)
    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.95, 1.05, n_customers), 2)

    support_calls = rng.poisson(1.2 + 1.3 * (internet == "fiber"), n_customers)
    senior = rng.random(n_customers) < 0.16
    paperless = rng.random(n_customers) < 0.6
    has_partner = rng.random(n_customers) < 0.48

    # Latent log-odds of churn built from known drivers.
    logit = (
        -0.1
        + 1.5 * (contract == "month-to-month")
        - 1.2 * (contract == "two-year")
        - 0.9 * np.log1p(tenure) + 1.0 * (tenure <= 3)  # non-linear: onboarding cliff
        + 0.55 * (internet == "fiber")
        + 0.6 * (payment == "electronic-check")
        + 0.25 * support_calls
        + 1.1 * ((internet == "fiber") & (support_calls >= 3))  # interaction: unhappy fiber users
        - 0.8 * ((contract == "month-to-month") & (num_products >= 4))  # bundled MTM users stay
        + 0.012 * (monthly_charges - 60)
        + 0.35 * senior
        - 0.15 * num_products
        - 0.3 * has_partner
        + 0.2 * paperless
        + rng.normal(0, 0.6, n_customers)
    )
    churned = rng.random(n_customers) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_customers)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_products": num_products,
            "support_calls_90d": support_calls,
            "is_senior": senior.astype(int),
            "paperless_billing": paperless.astype(int),
            "has_partner": has_partner.astype(int),
            "churned": churned.astype(int),
        }
    )
    # Real data is messy: inject ~1% missing total_charges (new accounts).
    missing = rng.random(n_customers) < 0.01
    df.loc[missing, "total_charges"] = np.nan
    return df
