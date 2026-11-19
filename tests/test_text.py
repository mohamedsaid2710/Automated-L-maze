from lmaze.text import copy_case, copy_punct, edit_distance, n_units, normalise, split_punct, units


def test_units_are_grapheme_clusters_in_any_script():
    assert n_units("Größe") == 5
    assert n_units(normalise("Größe")) == 5          # decomposed ö, normalised
    assert n_units("Größe") == 5                     # a grapheme even without NFC
    assert n_units("हिन्दी") == 2                                 # Devanagari: two aksharas, हि + न्दी (conjunct)
    assert n_units("كَتَبَ") == 3                                   # Arabic letters with harakat
    assert units("中文", "char") == ["中", "文"]


def test_punctuation_split_and_copy():
    assert split_punct("„Haus,“") == ("„", "Haus", ",“")
    assert copy_punct("geladen.", "buchker") == "buchker."
    assert copy_punct("(Tisch)", "Deten") == "(Deten)"
    assert split_punct("—") == ("—", "", "")
    # combining marks at the end belong to the word, not to the punctuation
    assert split_punct("الْكِتَابُ.")[1] == "الْكِتَابُ"


def test_copy_case():
    assert copy_case("Tischen", "regesch") == "Regesch"
    assert copy_case("waren", "kleit") == "kleit"
    assert copy_case("NDR", "ezu") == "EZU"
    assert copy_case("Straße", "ßtrabe") == "ßtrabe"            # ß has no one-letter capital: kept
    assert copy_case("Haus", "كتاب", has_case=False) == "كتاب"


def test_edit_distance():
    assert edit_distance("Tisch", "Tisch") == 0
    assert edit_distance(list("Tisch"), list("Fisch")) == 1
    assert edit_distance("", "abc") == 3
