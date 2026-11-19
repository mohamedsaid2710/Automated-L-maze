from pathlib import Path

import pytest

from lmaze.profile import load_profile, read_params

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("code", ["de", "en", "ar"])
def test_shipped_params_files_load(code):
    p = load_profile(ROOT / f"params_{code}.txt")
    assert p.language == code


def test_params_syntax(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text('# comment\nlanguage: "xx"   # inline\nlanguage_name: Test\nscript: "Latin"\n'
                 'has_case: false\nn_samples: 50 # fewer\ntrain_min_zipf: 3\n', encoding="utf-8")
    assert read_params(f)["has_case"] is False
    p = load_profile(f)
    assert (p.language, p.language_name, p.has_case, p.n_samples, p.train_min_zipf) == ("xx", "Test", False, 50, 3.0)


def test_params_errors(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text('language: "xx"\nlanguage_name: "T"\nscript: "Latin"\nn_sampels: 5\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown parameter"):
        load_profile(f)
    f.write_text('language: "xx"\nlanguage_name: "T"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_profile(f)
    f.write_text('language: "xx"\nlanguage_name: "T"\nscript: "Latin"\nn_samples: "many"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="n_samples"):
        load_profile(f)


def test_seed_is_none_or_a_whole_number(tmp_path):
    f = tmp_path / "p.txt"
    base = 'language: "xx"\nlanguage_name: "T"\nscript: "Latin"\n'
    for raw, want in (("None", None), ("7", 7)):
        f.write_text(base + f"seed: {raw}\n", encoding="utf-8")
        assert load_profile(f).seed == want
    f.write_text(base + "seed: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_profile(f)
