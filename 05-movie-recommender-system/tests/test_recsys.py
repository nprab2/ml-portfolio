import numpy as np
import pandas as pd

from recsys.data import generate_ratings, temporal_user_split
from recsys.models import BiasBaseline, BiasedALS, ImplicitALS, Popularity, ranking_metrics


def small():
    r, items = generate_ratings(n_users=200, n_items=150, avg_ratings_per_user=20, seed=1)
    return r, r["user_id"].max() + 1, len(items)


def test_split_is_temporal_and_disjoint():
    r, _, _ = small()
    tr, te = temporal_user_split(r)
    assert len(tr) + len(te) == len(r)
    last_train = tr.groupby("user_id")["timestamp"].max()
    first_test = te.groupby("user_id")["timestamp"].min()
    assert (last_train.loc[first_test.index] <= first_test).all()


def test_als_beats_bias_baseline_on_train_and_converges():
    r, nu, ni = small()
    bias = BiasBaseline().fit(r, nu, ni)
    als = BiasedALS(n_factors=8, factor_reg=0.1, n_iter=6).fit(r, nu, ni)
    u, i, y = r["user_id"].to_numpy(), r["item_id"].to_numpy(), r["rating"].to_numpy()
    assert np.mean((als.predict(u, i) - y) ** 2) < np.mean((bias.predict(u, i) - y) ** 2)
    assert als.history[-1] <= als.history[0]


def test_ranking_metrics_on_a_perfect_model():
    train = pd.DataFrame({"user_id": [0], "item_id": [0], "rating": [5]})
    test = pd.DataFrame({"user_id": [0], "item_id": [2], "rating": [5]})

    class Oracle:
        def score_all(self, users):
            return np.array([[9.0, 0.0, 1.0, 0.0]])   # item 0 is highest but already seen

    m = ranking_metrics(Oracle(), train, test, n_items=4, k=1)
    assert m["precision@1"] == 1.0 and m["ndcg@1"] == 1.0


def test_implicit_als_ranks_better_than_popularity_on_train_interactions():
    r, nu, ni = small()
    tr, te = temporal_user_split(r)
    ials = ranking_metrics(ImplicitALS(16, reg=20, alpha=10, n_iter=5).fit(tr, nu, ni), tr, te, ni)
    assert ials["catalog_coverage"] > ranking_metrics(Popularity().fit(tr, nu, ni), tr, te, ni)["catalog_coverage"]
