"""Reading the stimulus file:

    tag;id;sentence[;labels]

- separated by semicolons, tabs or commas (detected from the first line, in that order of preference,
  so commas inside sentences are fine in a semicolon or tab file); UTF-8;
- with a header row, columns are found by name (tag, id or item or item_id, sentence, labels), in any
  order, and other columns are ignored; without one, the columns are tag, id, sentence, labels in this
  order (a first row is a header only if it names an id and a sentence column);
- the sentence is split on whitespace only, punctuation stays attached to its word;
- labels (optional) give one label per word. Words with the same label in the rows of one item get
  the same distractor, so conditions differ only in their critical words (Latin square). Without
  labels, each position is its own label (0, 1, 2, ...).
"""

import csv
import logging
from dataclasses import dataclass, field

from .text import normalise, strip_punct


@dataclass
class Sentence:
    tag: str
    id: str
    word_sentence: str
    words: list
    labels: list
    distractors: list = field(default_factory=list)

    @property
    def distractor_sentence(self):
        return " ".join(self.distractors)

    @property
    def label_sentence(self):
        return " ".join(str(lab) for lab in self.labels)


@dataclass
class SentenceSet:
    id: str
    sentences: list = field(default_factory=list)


DELIMITERS = (";", "\t", ",")
COLUMNS = {"tag": ("tag",), "id": ("id", "item", "item_id"), "sentence": ("sentence",), "labels": ("labels",)}


def detect_delimiter(sample):
    """The first of ; tab , that splits the first line into at least three fields."""
    first = next((ln for ln in sample.splitlines() if ln.strip()), "")
    for d in DELIMITERS:
        if len(next(csv.reader([first], delimiter=d, quotechar='"'), [])) >= 3:
            return d
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        return next((d for d in DELIMITERS if d in sample), ";")


def tokenize(sentence):
    return sentence.split()


def _header_columns(row):
    """{column: index} if the row is a header (it names an id and a sentence column), else None."""
    names = [c.strip().lower() for c in row]
    found = {col: next((i for i, n in enumerate(names) if n in aliases), None) for col, aliases in COLUMNS.items()}
    if found["id"] is None or found["sentence"] is None:
        return None
    if found["tag"] is None:
        raise ValueError(f"Header {row}: no 'tag' column (needed: tag, id, sentence; labels optional)")
    return found


def read_input(path, norm="NFC"):
    """Read a stimulus file into {item id: SentenceSet}, in file order."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delim = detect_delimiter(text[:4096])
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delim, quotechar='"') if any(c.strip() for c in r)]
    cols = _header_columns(rows[0]) if rows else None
    first = 1
    if cols:
        rows, first = rows[1:], 2
    else:
        cols = {"tag": 0, "id": 1, "sentence": 2, "labels": 3}
    need = max(cols["tag"], cols["id"], cols["sentence"]) + 1
    items = {}
    for ln, row in enumerate(rows, start=first):
        if len(row) < need:
            raise ValueError(f"Line {ln}: expected at least {need} fields, got {len(row)} "
                             f"(separator {delim!r}; columns tag, id, sentence[, labels])")
        tag, item_id, sentence = row[cols["tag"]], row[cols["id"]].strip(), normalise(row[cols["sentence"]].strip(), norm)
        lab_cell = row[cols["labels"]].strip() if cols["labels"] is not None and cols["labels"] < len(row) else ""
        words = tokenize(sentence)
        if not words:
            raise ValueError(f"Line {ln}: empty sentence")
        empty = [w for w in words if not strip_punct(w)]
        if empty:
            logging.warning("Line %d: token(s) without letters (%s) still take a distractor slot", ln, empty)
        if lab_cell:
            labels = lab_cell.split()
            if len(labels) != len(words):
                raise ValueError(f"Line {ln}: {len(labels)} labels for {len(words)} words")
        else:
            labels = [str(i) for i in range(len(words))]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Line {ln}: duplicate labels in one sentence")
        items.setdefault(item_id, SentenceSet(item_id)).sentences.append(
            Sentence(tag, item_id, sentence, words, labels))
    return items
