import pytest

from lmaze.items import read_input
from lmaze.output import ibex_line, save_delim


def write(tmp_path, text):
    p = tmp_path / "in.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_semicolons_win_over_commas_inside_sentences(tmp_path):
    items = read_input(write(tmp_path, "a;1;Der Mann, der kam, lachte.\nb;1;Die Frau, die kam, lachte.\n"))
    assert list(items) == ["1"]
    assert len(items["1"].sentences) == 2
    assert items["1"].sentences[0].words == ["Der", "Mann,", "der", "kam,", "lachte."]


def test_header_is_skipped_and_numeric_first_row_is_data(tmp_path):
    assert list(read_input(write(tmp_path, "tag;id;sentence\nx;1;Das ist gut.\n"))) == ["1"]
    assert list(read_input(write(tmp_path, "x;1;Das ist gut.\nx;2;Das ist schlecht.\n"))) == ["1", "2"]


def test_tabs_and_commas(tmp_path):
    tab = read_input(write(tmp_path, "a\t1\tDer Mann, der kam, lachte.\t0 1 2 3 4\n"))
    assert tab["1"].sentences[0].words[1] == "Mann," and tab["1"].sentences[0].labels[4] == "4"
    comma = read_input(write(tmp_path, 'a,1,"Der Mann, der kam, lachte."\n'))
    assert comma["1"].sentences[0].words == ["Der", "Mann,", "der", "kam,", "lachte."]


def test_columns_by_header_name_in_any_order_extra_columns_ignored(tmp_path):
    text = "Sentence\tnotes\tItem\tlabels\ttag\nDas ist gut.\tx\t7\t0 1 2\tcond_a\n"
    s = read_input(write(tmp_path, text))["7"].sentences[0]
    assert (s.tag, s.words, s.labels) == ("cond_a", ["Das", "ist", "gut."], ["0", "1", "2"])
    with pytest.raises(ValueError, match="tag"):
        read_input(write(tmp_path, "id;sentence\n1;Das ist gut.\n"))


def test_labels(tmp_path):
    items = read_input(write(tmp_path, "a;1;Das ist gut.;0 1 2\nb;1;Das war gut.;0 9 2\n"))
    assert items["1"].sentences[1].labels == ["0", "9", "2"]
    assert read_input(write(tmp_path, "a;1;Das ist gut.\n"))["1"].sentences[0].labels == ["0", "1", "2"]
    with pytest.raises(ValueError, match="labels"):
        read_input(write(tmp_path, "a;1;Das ist gut.;0 1\n"))
    with pytest.raises(ValueError, match="duplicate"):
        read_input(write(tmp_path, "a;1;Das ist gut.;0 1 1\n"))


def test_input_is_nfc_normalised(tmp_path):
    items = read_input(write(tmp_path, "a;1;Größe Stadt.\n"))
    assert items["1"].sentences[0].words[0] == "Größe"


def test_ibex_line_escapes_quotes_and_backslashes():
    line = ibex_line("t", "1", 'Er sagte "ja" \\ gut.', "x-x-x Blum")
    assert line == '[["t", \'1\'], "Maze", {s:"Er sagte \\"ja\\" \\\\ gut.", a:"x-x-x Blum"}], \n'


def test_delim_output_columns(tmp_path):
    items = read_input(write(tmp_path, "a;1;Das ist gut.\n"))
    items["1"].sentences[0].distractors = ["x-x-x", "äng", "zud."]
    out = tmp_path / "out.txt"
    save_delim(out, items)
    assert out.read_text(encoding="utf-8").strip() == "a;1;Das ist gut.;x-x-x äng zud.;0 1 2"
