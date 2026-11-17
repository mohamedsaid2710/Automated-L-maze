"""Reading the stimulus file:

    tag;id;sentence[;labels]

- semicolon-delimited (comma only if the first line has fewer than three semicolon fields);
- UTF-8; an optional header row is skipped (a first row whose id is a number is data, not a header);
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


def detect_delimiter(sample):
    first = next((ln for ln in sample.splitlines() if ln.strip()), "")
    if first and len(next(csv.reader([first], delimiter=";", quotechar='"'), [])) >= 3:
        return ";"
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
    except csv.Error:
        return ";" if ";" in sample else ","


def tokenize(sentence):
    return sentence.split()


def _looks_like_header(row):
    return len(row) >= 2 and not row[1].strip().isdigit() and row[1].strip().lower() in ("id", "item", "item_id")


def read_input(path, norm="NFC"):
    """Read a stimulus file into {item id: SentenceSet}, in file order."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delim = detect_delimiter(text[:4096])
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delim, quotechar='"') if any(c.strip() for c in r)]
    if rows and _looks_like_header(rows[0]):
        rows = rows[1:]
    items = {}
    for ln, row in enumerate(rows, start=1):
        if len(row) < 3:
            raise ValueError(f"Line {ln}: expected at least 3 fields (tag;id;sentence), got {len(row)}")
        tag, item_id, sentence = row[0], row[1].strip(), normalise(row[2].strip(), norm)
        words = tokenize(sentence)
        if not words:
            raise ValueError(f"Line {ln}: empty sentence")
        empty = [w for w in words if not strip_punct(w)]
        if empty:
            logging.warning("Line %d: token(s) without letters (%s) still take a distractor slot", ln, empty)
        if len(row) > 3 and row[3].strip():
            labels = row[3].split()
            if len(labels) != len(words):
                raise ValueError(f"Line {ln}: {len(labels)} labels for {len(words)} words")
        else:
            labels = [str(i) for i in range(len(words))]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Line {ln}: duplicate labels in one sentence")
        items.setdefault(item_id, SentenceSet(item_id)).sentences.append(
            Sentence(tag, item_id, sentence, words, labels))
    return items
