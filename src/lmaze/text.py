"""Script-agnostic text helpers: normalisation, units, punctuation, case.

Nothing here assumes Latin letters, spaces inside words, case, or a writing direction. A word is
split into *units* (grapheme clusters by default, so a letter with its combining marks, a
Devanagari akshara with its vowel sign, or an Arabic letter with its harakat is one unit), and
everything that counts or compares length works on units.
"""

import unicodedata

import regex

_GRAPHEME = regex.compile(r"\X")


def normalise(text, form="NFC"):
    """Unicode-normalise, so that a precomposed and a decomposed ü compare equal."""
    return unicodedata.normalize(form, text) if form else text


def units(word, unit="grapheme"):
    """Split a word into units: 'grapheme' (Unicode extended grapheme clusters) or 'char'."""
    if unit == "grapheme":
        return _GRAPHEME.findall(word)
    if unit == "char":
        return list(word)
    raise ValueError(f"Unknown unit: {unit!r} (expected 'grapheme' or 'char')")


def n_units(word, unit="grapheme"):
    return len(units(word, unit))


def _is_word_char(ch):
    """Letters, digits and combining marks (Mn, Mc) belong to the word; the rest is punctuation.

    Combining marks count as word characters, otherwise a vowel sign or diacritic at the end of a
    word (Arabic harakat, Devanagari matras) would be stripped as punctuation.
    """
    return ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc")


def word_span(token):
    """(start, end) indices of the word body inside a token, or None if it has none."""
    start = next((i for i, ch in enumerate(token) if _is_word_char(ch)), None)
    if start is None:
        return None
    end = next(j for j in range(len(token) - 1, start - 1, -1) if _is_word_char(token[j]))
    return start, end


def split_punct(token):
    """Split a token into (leading punctuation, word body, trailing punctuation)."""
    span = word_span(token)
    if span is None:
        return token, "", ""
    start, end = span
    return token[:start], token[start:end + 1], token[end + 1:]


def strip_punct(token):
    return split_punct(token)[1]


def copy_punct(target_token, distractor):
    """Put the target token's leading and trailing punctuation around the distractor."""
    lead, _body, trail = split_punct(target_token)
    return lead + distractor + trail


def copy_case(target, distractor, has_case=True):
    """Give the distractor the case pattern of the target (scripts with case only).

    All capitals (and longer than one letter) -> all capitals; first letter capital -> first letter
    capital; otherwise lower case. Upper-casing never changes the length: a letter whose capital is
    longer (German ß -> SS) is left as it is.
    """
    if not has_case or not distractor:
        return distractor

    def up(ch):
        u = ch.upper()
        return u if len(u) == len(ch) else ch

    letters = [c for c in target if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return "".join(up(c) for c in distractor)
    if letters and target[0].isupper():
        first = units(distractor)[0]
        return "".join(up(c) for c in first) + distractor[len(first):]
    return distractor


def edit_distance(a, b):
    """Levenshtein distance between two unit sequences (lists or strings)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
