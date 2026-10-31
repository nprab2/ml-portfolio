# Machine Learning & Data Science Portfolio

Ten end-to-end Python projects spanning classical ML, deep learning from scratch, recommender systems, NLP, time-series forecasting, experimentation and predictive maintenance. Every project is **runnable in one command**, **unit-tested**, and documents real results with figures.

| # | Project | Problem type | Headline result | Key techniques |
|---|---|---|---|---|
| 01 | [Customer Churn Prediction](01-customer-churn-prediction) | Binary classification | +20% campaign profit vs. a 0.5 threshold | Calibrated gradient boosting, profit-optimal thresholding, permutation importance |
| 02 | [Credit Card Fraud Detection](02-credit-card-fraud-detection) | Imbalanced classification (0.6% fraud) | PR-AUC 0.88; 85% of fraud $ caught reviewing 0.5% of traffic | Leak-free behavioural features, out-of-time validation, Isolation Forest |
| 03 | [Retail Demand Forecasting](03-retail-demand-forecasting) | Time-series forecasting | 33% lower WAPE than seasonal naive; well-calibrated P10–P90 | Global gradient-boosting model, direct multi-horizon, quantile regression, rolling backtest |
| 04 | [Customer Segmentation](04-customer-segmentation) | Unsupervised clustering | Bootstrap stability ARI 0.995 | RFM+ features, empirical-Bayes shrinkage, k selection, cluster validation |
| 05 | [Movie Recommender System](05-movie-recommender-system) | Recommendation | NDCG@10 +64% vs. popularity | ALS & implicit ALS from scratch, item-kNN, ranking metrics |
| 06 | [Review Sentiment Analysis](06-nlp-review-sentiment) | NLP classification | 99.5% vs. true labels; 100% on negation stress test | TF-IDF word+char n-grams, negation scope marking, CheckList-style tests |
| 07 | [House Price Prediction](07-house-price-prediction) | Regression | R² 0.956; 90.7% coverage of 90% intervals | Stacking, target encoding, splines, conformal prediction |
| 08 | [A/B Testing Framework](08-ab-testing-framework) | Statistical inference | Naive peeking → 32% false positives; mSPRT → 0.4% | Power analysis, CUPED, Bayesian A/B, sequential testing, SRM |
| 09 | [Predictive Maintenance (RUL)](09-predictive-maintenance-rul) | Time-series regression | RMSE 18.8 cycles; 100% recall on near-failure engines | Regime normalisation, causal window features, NASA scoring |
| 10 | [Neural Networks from Scratch](10-neural-network-from-scratch) | Deep learning | CNN 98.1% on real handwritten digits | NumPy MLP & CNN (im2col), AdamW, gradient checking |

## Running any project

```bash
cd 01-customer-churn-prediction
pip install -r requirements.txt
python train.py          # (project 08 uses run_analysis.py)
python -m pytest -q
```

All projects need only `numpy`, `pandas`, `scikit-learn`, `scipy` and `matplotlib`, and run on a laptop CPU in under two minutes.

## About the data

Project 10 uses a real public dataset (UCI handwritten digits, bundled with scikit-learn). The other nine use **realistic simulators** written for each problem – customer behaviour with known churn drivers, transaction streams with fraud bursts, engine degradation physics, and so on. This is deliberate:

- Everything runs offline and reproducibly with fixed seeds.
- Because the ground truth is known, each project can *verify* its own conclusions (e.g., segmentation recovers the hidden personas; the churn model's top drivers match the generator; the conformal intervals hit their target coverage).

Each README explains how to plug in a well-known public dataset (IBM Telco, Sparkov, M5, Online Retail II, MovieLens, Ames Housing, NASA C-MAPSS, etc.).

## Engineering practices across the portfolio

- Leakage-safe pipelines (scikit-learn `Pipeline`, time-based and group-based splits, tests that tamper with future data)
- Baselines in every project, so improvements are measured, not assumed
- Business-relevant metrics (profit, dollars saved, WAPE, NASA score) alongside standard ones
- Continuous integration: GitHub Actions runs every project's tests on each push
