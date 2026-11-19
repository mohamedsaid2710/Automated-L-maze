"""Writing results: delim and ibex (PCIbex) formats, plus a report sidecar."""

import csv
import json
from pathlib import Path


def ibex_line(tag, item_id, sentence, distractors):
    """One PCIbex Maze item line (json.dumps escapes quotes, backslashes and newlines)."""
    s = json.dumps(sentence, ensure_ascii=False)
    d = json.dumps(distractors, ensure_ascii=False)
    t = json.dumps(tag, ensure_ascii=False)
    return f'[[{t}, {item_id!r}], "Maze", {{s:{s}, a:{d}}}], \n'


def save_delim(path, items):
    """tag;id;sentence;distractors;labels, one row per input row."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        for item in items.values():
            for s in item.sentences:
                w.writerow([s.tag, s.id, s.word_sentence, s.distractor_sentence, s.label_sentence])


def save_ibex(path, items):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for item in items.values():
            for s in item.sentences:
                f.write(ibex_line(s.tag, s.id, s.word_sentence, s.distractor_sentence))


REPORT_COLUMNS = [
    "seed", "item_id", "label", "position", "target", "distractor", "generator",
    "target_units", "distractor_units", "len_ok",
    "is_real_word", "edit_distance", "target_logprob", "distractor_logprob", "logprob_delta",
    "letter_prob_ratio",
    "n_candidates", "n_rejected_real_word", "relaxation", "is_placeholder", "note",
]


def report_path(path):
    """out.txt -> out.report.csv, next to the output (pathlib, so Windows paths work too)."""
    p = Path(path)
    return str(p.with_name(p.stem + ".report.csv"))


def save_report(path, rows):
    with open(report_path(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return report_path(path)
