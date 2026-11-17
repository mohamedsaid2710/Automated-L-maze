import random

from lmaze.ngram import BOS, EOS, NgramModel, PositionalModel
from lmaze.text import units

# A toy Hebrew lexicon: final letters (ך ם ן ף ץ) occur only at the end of a word.
HEBREW = ["שלום", "מלך", "ספר", "דרך", "לחם", "אדם", "כסף", "ארץ", "מים", "שמן", "ילד", "בית",
          "מלכה", "ספרים", "דרכים", "לחמים", "ילדים", "שמים", "כספים", "ארצות", "מלכים", "בתים"]
FINALS = set("ךםןףץ")


def model(words, min_count=1):
    return NgramModel([tuple(units(w)) for w in words], min_count=min_count)


def legal(m, seq):
    padded = m.start + tuple(seq) + (EOS,)
    return all(padded[i + m.k] in m.tri.get(padded[i:i + m.k], {}) for i in range(len(padded) - m.k))


def test_samples_have_exact_length_and_only_attested_trigrams():
    m = model(HEBREW)
    rng = random.Random(1)
    for n in (3, 4, 5):
        for _ in range(50):
            s = m.sample(n, rng)
            assert s is not None and len(s) == n and legal(m, s)


def test_hebrew_final_letters_only_at_the_end():
    m = model(HEBREW)
    rng = random.Random(2)
    for _ in range(200):
        s = m.sample(4, rng)
        assert not FINALS & set(s[:-1])


def test_impossible_length_returns_none():
    m = model(["ab", "abc"])
    assert m.sample(7, random.Random(0)) is None
    assert m.candidates(tuple("abcdefg"), seed=0)[0] == []


def test_candidates_are_deterministic_and_sorted():
    m = model(HEBREW)
    a, ta = m.candidates(tuple(units("ספרים")), seed=42)
    b, tb = m.candidates(tuple(units("ספרים")), seed=42)
    assert a == b and ta == tb
    deltas = [abs(lp - ta) for _, lp in a]
    assert deltas == sorted(deltas)


def test_min_count_removes_rare_trigrams_from_generation():
    m = model(["abc", "abc", "abd"], min_count=2)
    assert "d" not in m.tri[("a", "b")]
    assert m.counts[("a", "b")]["d"] == 1                     # still counted for probabilities


def test_positional_model_makes_new_two_character_combinations():
    # Chinese-like: any n-gram of a two-character word is the word itself, so the positional model
    # combines a word-initial with a word-final character, giving new combinations of real characters.
    words = ["学生", "学校", "先生", "医生", "校长", "医院"]
    m = PositionalModel([tuple(w) for w in words])
    made = {"".join(m.sample(2, random.Random(i))) for i in range(200)}
    assert made - set(words)
    starts, ends = {w[0] for w in words}, {w[-1] for w in words}
    assert all(w[0] in starts and w[1] in ends for w in made)
