"""Parameters: everything language-specific or tunable lives in one params file per language.

Format: one `key: value` per line, `#` starts a
comment (also after a value), strings in quotes, true/false/null spelled either way. Adding a
language = copying a params file and changing its values.
"""

import ast
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

_REQUIRED = ("language", "language_name", "script")


@dataclass(frozen=True)
class Profile:
    # --- Language ---
    language: str                      # ISO 639-1 code, also the wordfreq language code
    language_name: str
    script: str                        # Unicode script name ("Latin", "Arabic", "Devanagari",
                                       # "Han"): only words in this script train the generator
    direction: str = "ltr"             # "ltr" or "rtl" (documentation; the core ignores it)
    has_case: bool = True              # copy the target's capitalisation onto the distractor
    unit: str = "grapheme"             # length unit: "grapheme" or "char"
    normalise: str = "NFC"             # Unicode normalisation before any comparison
    lowercase: bool = True             # compare and model lowercased forms (scripts with case)
    strip_marks: bool = False          # drop combining marks before lookup (vocalised Arabic)
    # --- Lexicon (wordfreq) ---
    wordlist: str = "large"            # wordfreq list: "large" where available, else "best"
    train_min_zipf: float = 2.0        # only words at least this frequent train the generator
    train_max_words: int = 100_000
    alphabet: str = ""                 # if set, only training words made of these units
    vowels: str = ""                   # if set, a candidate must contain at least one of these units
    final_only: str = ""               # units allowed only as the last unit (Arabic ة ى)
    not_final: str = ""                # units never allowed as the last unit (e.g. non-final letter forms)
    # --- Generator ---
    n_samples: int = 400               # candidates sampled per target
    temperature: float = 1.0           # 1 = follow the model; > 1 gives rarer units more chance, < 1 fewer
    min_edit_distance: int = 2         # the distractor differs from the target in at least this many
                                       # letters (letters to change, add or delete: Levenshtein distance)
    len_tolerance: int = 1             # if a word has no same-length pseudoword (short words), allow
                                       # this many units shorter or longer; 0 = never (placeholder)
    max_repeat: int = 1                # how often one distractor may be used in a whole run (one label
                                       # shared by an item's conditions counts once); 0 = no limit
    short_word_max_len: int = 3        # distractors this short (in units) count as short words
    max_repeat_short: int = 3          # how often one short distractor may be used in a whole run
    min_word_distance: int = 2         # a (non-short) distractor must differ from every common word
                                       # (Zipf >= train_min_zipf) in at least this many letters; 0/1 = off
    near_word_min_zipf: float = 0.0    # the common words for min_word_distance: Zipf >= this; 0 = train_min_zipf
    prefixes: str = ""                 # (non-short) distractors that are a common word plus one of
    suffixes: str = ""                 # these prefixes and/or up to two of these suffixes are rejected
                                       # (Arabic تكملتك = تكملة + ك); space separated
    stem_end_changes: str = ""         # "a>b ...": before a suffix a stem-final b is written a
                                       # (Arabic "ت>ة": تكملت + ك -> تكملة); ">b" = b dropped
                                       # (Arabic ">ة": رياض + ات -> رياضة)
    # --- Output ---
    first_word: int = 1                # 1 = x-x-x at the first position; 0 = a pseudoword, like every other word
    seed: Optional[int] = None         # None = random each run (printed and in the report); a number = same output each run
    write_report: bool = True          # also write <output>.report.csv

    @property
    def code(self):
        return self.language

    @property
    def name(self):
        return self.language_name


_FIELDS = {f.name: f for f in fields(Profile)}


def _parse_value(raw):
    """One value: a Python literal (inline `# comment` allowed), true/false/null, or a bare string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    stripped = raw.split("#", 1)[0].strip()
    low = stripped.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped.strip('"')


def read_params(path):
    """Read a params file into a dict (key: value lines; blank lines and # comments ignored)."""
    params = {}
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            logging.warning("Skipping malformed params line %d in %s: %r", lineno, path, line)
            continue
        key, raw = line.split(":", 1)
        params[key.strip()] = _parse_value(raw.strip())
    return params


def load_profile(path):
    """Read and check a params file: unknown keys, missing required keys and wrong types are errors."""
    params = read_params(path)
    unknown = set(params) - set(_FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown parameter(s): {', '.join(sorted(unknown))}")
    missing = [k for k in _REQUIRED if k not in params]
    if missing:
        raise ValueError(f"{path}: missing required parameter(s): {', '.join(missing)}")
    for key, value in params.items():
        if key == "seed":
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{path}: seed should be a whole number or None, got {value!r}")
            continue
        want = str if key in _REQUIRED else type(_FIELDS[key].default)
        if want is float and isinstance(value, int) and not isinstance(value, bool):
            params[key] = float(value)
        elif not isinstance(value, want) or (want is int and isinstance(value, bool)):
            raise ValueError(f"{path}: {key} should be {want.__name__}, got {value!r}")
    return Profile(**params)
