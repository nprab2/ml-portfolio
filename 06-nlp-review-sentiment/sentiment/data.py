"""Compositional product-review generator.

Reviews are built from aspects, opinion words, negation, intensifiers and
contrastive "but" clauses, with typos and label noise. Because the grammar is
known, we can also build *challenge sets* that probe specific linguistic skills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASPECTS = ["battery life", "screen", "shipping", "customer service", "price", "sound quality",
           "build quality", "camera", "setup process", "packaging", "app", "keyboard"]
POS = ["great", "excellent", "amazing", "solid", "fantastic", "reliable", "impressive", "smooth",
       "perfect", "good", "decent", "outstanding", "sturdy", "fast"]
NEG = ["terrible", "awful", "poor", "disappointing", "flimsy", "slow", "broken", "useless",
       "bad", "horrible", "mediocre", "frustrating", "cheap", "laggy"]
INTENS = ["really", "very", "extremely", "incredibly", "super", "pretty", ""]
NEGATORS = ["not", "never", "hardly", "not at all"]
OPENERS = ["", "Honestly, ", "Overall, ", "Update: ", "After two weeks, ", "For the money, "]
POS_CLOSERS = ["Would buy again.", "Highly recommend.", "Five stars.", "Happy with it.", ""]
NEG_CLOSERS = ["Returning it.", "Would not recommend.", "Waste of money.", "Avoid.", ""]


def _clause(rng, positive: bool, allow_negation: bool = True) -> str:
    aspect = rng.choice(ASPECTS)
    intens = rng.choice(INTENS)
    if allow_negation and rng.random() < 0.25:
        # "the screen is not good" -> negative; "the screen is not bad" -> positive
        word = rng.choice(NEG if positive else POS)
        return f"the {aspect} is {rng.choice(NEGATORS)} {word}"
    word = rng.choice(POS if positive else NEG)
    return f"the {aspect} is {intens + ' ' if intens else ''}{word}"


def _typo(rng, text: str, rate: float) -> str:
    chars = list(text)
    for j in range(len(chars) - 1):
        if chars[j].isalpha() and rng.random() < rate:
            chars[j], chars[j + 1] = chars[j + 1], chars[j]
    return "".join(chars)


def make_review(rng, positive: bool, typo_rate: float = 0.01) -> str:
    n = rng.integers(1, 4)
    clauses = [_clause(rng, positive) for _ in range(n)]
    if rng.random() < 0.3:
        # Contrast: an opposite-polarity clause *before* "but" – the final clause dominates.
        clauses.insert(0, _clause(rng, not positive, allow_negation=False) + ", but")
    text = rng.choice(OPENERS) + " ".join(c if c.endswith("but") else c + "." for c in clauses)
    closer = rng.choice(POS_CLOSERS if positive else NEG_CLOSERS) if rng.random() < 0.5 else ""
    text = (text + " " + closer).strip()
    text = text[0].upper() + text[1:]
    return _typo(rng, text, typo_rate)


def generate_reviews(n: int = 20_000, label_noise: float = 0.03, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.5
    texts = [make_review(rng, bool(y)) for y in labels]
    flip = rng.random(n) < label_noise
    # ``label`` is what an annotator wrote (noisy); ``clean_label`` is the true sentiment,
    # kept only to measure how close models get to the noise ceiling.
    return pd.DataFrame({"text": texts, "label": (labels ^ flip).astype(int),
                         "clean_label": labels.astype(int)})


def challenge_sets(n: int = 500, seed: int = 99) -> dict[str, pd.DataFrame]:
    """Targeted test suites in the spirit of CheckList (Ribeiro et al., 2020)."""
    rng = np.random.default_rng(seed)
    sets = {}
    rows = []
    for _ in range(n):
        y = rng.random() < 0.5
        a = rng.choice(ASPECTS)
        w = rng.choice(NEG if y else POS)
        rows.append((f"The {a} is {rng.choice(NEGATORS)} {w}.", int(y)))
    sets["negation"] = pd.DataFrame(rows, columns=["text", "label"])
    rows = []
    for _ in range(n):
        y = rng.random() < 0.5
        first = _clause(rng, not y, allow_negation=False)
        second = _clause(rng, bool(y), allow_negation=False)
        rows.append((f"{first.capitalize()}, but {second}.", int(y)))
    sets["contrast_but"] = pd.DataFrame(rows, columns=["text", "label"])
    rows = []
    for _ in range(n):
        y = rng.random() < 0.5
        rows.append((make_review(rng, bool(y), typo_rate=0.08), int(y)))
    sets["heavy_typos"] = pd.DataFrame(rows, columns=["text", "label"])
    return sets
