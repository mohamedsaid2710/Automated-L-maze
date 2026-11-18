"""The language's word list: what trains the generator, and what a distractor must not be.

Lookups go straight to wordfreq's frequency table (no tokenizer), so they work the same for every
language, including Chinese, where wordfreq's own lookup would need the jieba segmenter.
"""

import math
import unicodedata
from functools import lru_cache

import regex
import wordfreq

from .text import normalise, units


@lru_cache(maxsize=8)
def _freq_table(lang, wordlist):
    return wordfreq.get_frequency_dict(lang, wordlist=wordlist)


@lru_cache(maxsize=16)
def _script_pattern(script):
    """A word is in the profile's script: letters of that script plus combining marks."""
    return regex.compile(rf"^(?:\p{{Script={script}}}|\p{{M}})+$") if script else None


def strip_marks(text):
    """Remove combining marks (Arabic harakat, Hebrew points): wordfreq lists unvocalised forms."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn"))


class Lexicon:
    def __init__(self, profile):
        self.profile = profile
        self.lang = profile.code
        if self.lang not in wordfreq.available_languages(wordlist=profile.wordlist):
            raise ValueError(f"wordfreq has no {profile.wordlist!r} list for {self.lang!r}")
        self.table = _freq_table(self.lang, profile.wordlist)

    def key(self, word):
        """The form used for lookup and modelling: normalised; lowercased for scripts with case;
        without combining marks where the profile says so (vocalised Arabic or Hebrew input)."""
        form = normalise(word, self.profile.normalise)
        if self.profile.strip_marks:
            form = strip_marks(form)
        return form.lower() if self.profile.lowercase else form

    def zipf(self, word):
        f = self.table.get(self.key(word), 0.0)
        return round(math.log10(f) + 9, 2) if f > 0 else 0.0

    def is_real_word(self, word):
        """Constraint 2: the string is in the word list (any entry counts)."""
        return self.key(word) in self.table

    def common_words(self):
        """Words of the list at least as frequent as the training floor (train_min_zipf): the words a
        reader knows. Web noise (typos, names, foreign strings) stays out."""
        if not hasattr(self, "_common"):
            floor = 10 ** (self.profile.train_min_zipf - 9)
            self._common = {self.key(w) for w, f in self.table.items() if f >= floor}
        return self._common

    def near_common_word(self, cand_units, alphabet):
        """True if one edit (change, add or delete one unit) turns the candidate into a common word.
        Such a string is read as a misspelling of that word rather than as a nonword (Arabic
        اننتصب is read as انتصب)."""
        common = self.common_words()
        u = list(cand_units)
        for i in range(len(u) + 1):
            if i < len(u):
                if "".join(u[:i] + u[i + 1:]) in common:                       # delete
                    return True
            for a in alphabet:
                if "".join(u[:i] + [a] + u[i:]) in common:                     # add
                    return True
                if i < len(u) and a != u[i] and "".join(u[:i] + [a] + u[i + 1:]) in common:   # change
                    return True
        return False

    def training_words(self):
        """Words that train the n-gram model, as unit tuples (frequent, clean words only)."""
        p = self.profile
        return _training_words(self.lang, p.wordlist, p.train_min_zipf, p.train_max_words, p.unit,
                               p.normalise, p.lowercase, p.strip_marks, p.alphabet, p.script, p.vowels)


@lru_cache(maxsize=8)
def _training_words(lang, wordlist, min_zipf, max_words, unit, norm, lowercase, marks, alphabet, script,
                    vowels=""):
    allowed = set(units(normalise("".join(alphabet.split()), norm), unit)) if alphabet else None
    vowel_set = set(units("".join(vowels.split()), unit)) if vowels else None
    in_script = _script_pattern(script)
    min_freq = 10 ** (min_zipf - 9)
    out = []
    for w, f in _freq_table(lang, wordlist).items():      # sorted by frequency, most frequent first
        if f < min_freq or len(out) >= max_words:
            break
        form = normalise(w, norm)
        if marks:
            form = strip_marks(form)
        if lowercase:
            form = form.lower()
        if in_script and not in_script.match(form):
            continue                                       # digits, Latin words in a Hebrew list, ...
        u = tuple(units(form, unit))
        if allowed is not None and not all(x in allowed for x in u):
            continue
        if vowel_set is not None and not vowel_set & set(u):
            continue                                       # acronyms (npd, spd): no vowel, bad letter statistics
        out.append(u)
    return tuple(out)
