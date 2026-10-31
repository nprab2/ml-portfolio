"""Sentiment models from a lexicon baseline to linguistically-informed TF-IDF pipelines."""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import NEG, POS

NEGATION_RE = re.compile(r"\b(not|never|hardly|no|n't)\b", re.I)
TOKEN_RE = re.compile(r"[a-z']+|[.,!?;]")


def mark_scopes(text: str) -> str:
    """Linguistic preprocessing.

    * Negation scope: tokens after a negator get a ``NOT_`` prefix until punctuation
      ("is not good." -> "is not NOT_good .").
    * Contrast scope: tokens before "but" in a sentence get a ``PRE_`` prefix, letting the
      model learn that the clause after "but" carries the verdict.
    """
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.lower()):
        tokens = TOKEN_RE.findall(sentence)
        if "but" in tokens:
            cut = tokens.index("but")
            tokens = [f"PRE_{t}" if t.isalpha() else t for t in tokens[:cut]] + tokens[cut:]
        negating = False
        for t in tokens:
            bare = t.replace("PRE_", "")
            if bare in {".", ",", "!", "?", ";", "but"}:
                negating = False
                out.append(t)
            elif NEGATION_RE.fullmatch(bare):
                negating = True
                out.append(t)
            else:
                out.append(f"NOT_{t}" if negating else t)
    return " ".join(out)


class ScopeMarker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [mark_scopes(t) for t in X]


class LexiconBaseline(BaseEstimator, ClassifierMixin):
    """Count positive minus negative opinion words. No learning, no negation handling."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        pos, neg = set(POS), set(NEG)
        return np.array([sum(w in pos for w in TOKEN_RE.findall(t.lower()))
                         - sum(w in neg for w in TOKEN_RE.findall(t.lower())) for t in X], float)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def _clf():
    return LogisticRegression(C=4.0, max_iter=3000)


def build_models() -> dict:
    return {
        "lexicon_baseline": LexiconBaseline(),
        "tfidf_unigram_lr": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
            ("clf", _clf())]),
        "tfidf_word1-3_char2-5_lr": Pipeline([
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                         sublinear_tf=True, max_features=150_000))])),
            ("clf", _clf())]),
        "scope_marking + word/char_lr": Pipeline([
            ("scope", ScopeMarker()),
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True,
                                         token_pattern=r"[A-Za-z_']+", lowercase=False)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                         sublinear_tf=True, max_features=150_000))])),
            ("clf", _clf())]),
    }


def top_features(pipeline: Pipeline, n: int = 12) -> dict[str, list[str]]:
    """Most positive/negative word-level n-grams of a linear text pipeline."""
    clf = pipeline.named_steps["clf"]
    if "feats" in pipeline.named_steps:
        names = pipeline.named_steps["feats"].get_feature_names_out()
    else:
        names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coef = clf.coef_.ravel()
    word = np.array([not n.startswith("char__") for n in names])
    idx = np.flatnonzero(word)
    order = idx[np.argsort(coef[idx])]
    clean = lambda s: s.replace("word__", "")
    return {"positive": [clean(names[i]) for i in order[::-1][:n]],
            "negative": [clean(names[i]) for i in order[:n]]}
