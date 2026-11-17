"""End to end on German (needs wordfreq's German data, installed with wordfreq)."""

from pathlib import Path

import pytest

from lmaze.generate import Generator
from lmaze.items import read_input
from lmaze.profile import load_profile
from lmaze.text import n_units, strip_punct

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PARAMS_DE = ROOT / "params_de.txt"


@pytest.fixture(scope="module")
def german():
    return Generator(load_profile(PARAMS_DE), seed=0)


@pytest.fixture(scope="module")
def run(german):
    items = read_input(ROOT / "tests" / "data" / "german_items.txt")
    rows = german.run(items)
    return items, rows


def test_every_word_gets_a_matched_distractor(run):
    items, rows = run
    for item in items.values():
        for s in item.sentences:
            assert len(s.distractors) == len(s.words)
            assert s.distractors[0] == "x-x-x"
            for w, d in zip(s.words[1:], s.distractors[1:]):
                diff = n_units(strip_punct(d)) - n_units(strip_punct(w))
                assert diff == 0 or (abs(diff) == 1 and n_units(strip_punct(w)) <= 3), (w, d)
    for r in rows:                                      # another length is always reported
        assert r["len_ok"] or r["relaxation"].startswith("length differs") or r["is_placeholder"], r


def test_strict_distractors_are_not_words(run, german):
    _items, rows = run
    for r in rows:
        if not r["relaxation"] and not r["is_placeholder"]:
            assert not german.lexicon.is_real_word(r["distractor"]), r


def test_case_and_punctuation_follow_the_target(run):
    items, _rows = run
    s = items["1"].sentences[0]                       # Das Gewehr auf dem Tisch ... geladen.
    assert s.distractors[1][0].isupper()              # Gewehr (noun)
    assert s.distractors[2][0].islower()              # auf
    assert s.distractors[-1].endswith(".")


def test_same_label_same_distractor_across_conditions(run):
    items, _rows = run
    rows = items["1"].sentences
    # Gewehr (label 1) and geladen (label 10) are shared by all four conditions
    assert len({s.distractors[1] for s in rows}) == 1
    assert len({s.distractors[10] for s in rows}) == 1
    # war vs. waren (labels 7 and 70) get different distractors
    assert rows[0].distractors[7] != rows[1].distractors[7]


def test_no_distractor_twice_within_an_item(run):
    items, _rows = run
    for item in items.values():
        by_label = {}
        for s in item.sentences:
            for lab, d in zip(s.labels[1:], s.distractors[1:]):
                by_label[lab] = strip_punct(d).lower()
        assert len(set(by_label.values())) == len(by_label)


def test_same_seed_same_output(german):
    a = read_input(EXAMPLES / "german_sample.txt")
    b = read_input(EXAMPLES / "german_sample.txt")
    german.run(a)
    Generator(load_profile(PARAMS_DE), seed=0).run(b)
    assert [s.distractors for i in a.values() for s in i.sentences] == \
           [s.distractors for i in b.values() for s in i.sentences]


def test_hyphenated_compound_keeps_hyphen_and_single_letters(german):
    pw, rep = german.pseudoword("U-Boot", set())
    assert pw is not None and pw.startswith("U-") and n_units(pw) == 6
    pw, rep = german.pseudoword("Max-Planck", set())
    assert pw is not None and pw.count("-") == 1 and pw[0].isupper() and pw.split("-")[1][0].isupper()
