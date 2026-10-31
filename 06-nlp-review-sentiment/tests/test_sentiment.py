from sentiment.data import challenge_sets, generate_reviews
from sentiment.model import LexiconBaseline, build_models, mark_scopes


def test_negation_scope_stops_at_punctuation():
    out = mark_scopes("The screen is not good. The app is great.")
    assert "NOT_good" in out
    assert "NOT_great" not in out and "great" in out


def test_contrast_scope_marks_clause_before_but():
    out = mark_scopes("The price is awful, but the camera is amazing.")
    assert "PRE_awful" in out and "amazing" in out and "PRE_amazing" not in out


def test_lexicon_baseline_fails_negation_by_design():
    assert LexiconBaseline().fit([]).predict(["The screen is not good."])[0] == 1


def test_best_model_learns_negation():
    df = generate_reviews(3000, seed=3)
    model = build_models()["scope_marking + word/char_lr"].fit(df["text"].tolist(), df["label"])
    neg = challenge_sets(100)["negation"]
    acc = (model.predict(neg["text"].tolist()) == neg["label"]).mean()
    assert acc > 0.9
