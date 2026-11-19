"""The letter model on a toy lexicon (trained in a few seconds, not cached)."""

import math

import pytest

from lmaze.neural import NeuralModel
from lmaze.text import units

HEBREW = ["שלום", "מלך", "ספר", "דרך", "לחם", "אדם", "כסף", "ארץ", "מים", "שמן", "ילד", "בית",
          "מלכה", "ספרים", "דרכים", "לחמים", "ילדים", "שמים", "כספים", "ארצות", "מלכים", "בתים"]


@pytest.fixture(scope="module")
def model():
    words = [tuple(units(w)) for w in HEBREW] * 20
    return NeuralModel(words, arch={"emb": 16, "hidden": 32, "dropout": 0.0},
                       train={"max_epochs": 40, "patience": 40, "batch": 32, "heldout": 0.05}, cache=False)


def test_samples_have_exact_length_and_known_letters(model):
    letters = {u for w in HEBREW for u in units(w)}
    for n in (2, 3, 5, 7):
        seqs = model.sample(n, 50, seed=1)
        assert len(seqs) == 50
        assert all(len(s) == n and set(s) <= letters for s in seqs)


def test_same_seed_same_samples(model):
    assert model.sample(4, 20, seed=3) == model.sample(4, 20, seed=3)


def test_scores_are_mean_log2_probabilities(model):
    a, b = model.score([tuple(units("ספרים")), tuple(units("םירפס"))])
    assert -math.log2(len(model.itos)) - 20 < b < a < 0       # a real word beats its mirror image


def test_candidates_sorted_by_distance_to_target(model):
    cands, target_lp = model.candidates(tuple(units("ספרים")), seed=1, n_samples=100)
    gaps = [abs(lp - target_lp) for _s, lp in cands]
    assert cands and gaps == sorted(gaps) and all(len(s) == 5 for s, _ in cands)
