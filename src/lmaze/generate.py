"""The pipeline: one pseudoword per label of every item, checked against the constraints.

Constraints applied here (numbers as in the plan; where a word has no strict solution, the
relaxations in Generator.RELAXATIONS are tried in order and reported):
1  same length as the target, in the profile's units (by construction of the generator)
2  not a real word (lexicon check)
4  (partly) at least `min_edit_distance` units away from the target
5  the target's capitalisation (scripts with case)
6  the target's leading and trailing punctuation
7  no distractor twice within an item; across the whole run at most `max_repeat` times, short
   distractors (up to `short_word_max_len` units) at most `max_repeat_short` times
   function words recur in every sentence, and the stock of short pseudowords is small)
8  position 0 gets the placeholder (x-x-x) if first_word is 1, else a pseudoword like any word
Labels: one distractor per label, shared by all rows of an item (Latin-square consistency).
"""

import logging
import zlib

import regex

from .lexicon import Lexicon
from .neural import NeuralModel
from .text import copy_case, copy_punct, edit_distance, strip_punct, units

PLACEHOLDER = "x-x-x"                  # first position (first_word: 1) and words with no pseudoword

# Inside a word body, anything that is not a letter or combining mark separates segments
# (hyphenated compounds, apostrophes): each segment gets its own pseudoword, the separators stay.
_SEGMENTS = regex.compile(r"([^\p{L}\p{M}]+)")


class Generator:
    def __init__(self, profile, seed=0):
        self.profile = profile
        self.seed = seed
        self.lexicon = Lexicon(profile)
        words = self.lexicon.training_words()
        self.model = NeuralModel(words, key_parts=(profile.language, profile.wordlist, profile.train_min_zipf,
                                                   profile.train_max_words, profile.alphabet, profile.vowels,
                                                   profile.unit, profile.normalise, profile.lowercase,
                                                   profile.strip_marks, profile.script),
                                 final_only=units("".join(profile.final_only.split()), profile.unit),
                                 not_final=units("".join(profile.not_final.split()), profile.unit))
        self.times_used = {}                # distractor -> how often used so far in this run
        self.unit_set = sorted({u for w in self.lexicon.training_words() for u in w})
        self.vowels = set(units("".join(profile.vowels.split()), profile.unit)) if profile.vowels else None

    def _seed(self, *parts):
        return zlib.crc32("|".join(map(str, (self.seed, *parts))).encode("utf-8"))

    # Relaxation steps, tried in order until a candidate passes. Short words (1-3 units) often have
    # no strict solution: almost every legal short string is listed somewhere in a large web
    # lexicon. Each relaxation is recorded in the report.
    #   (name, min edit distance or None = params value, length difference)
    RELAXATIONS = (
        ("", None, 0),                                        # strict
        ("edit distance 1", 1, 0),                            # may share all but one letter with the target
        ("length", None, 1),                            # len_tolerance: 1, 2, ... units shorter
                                                              # or longer, whichever matches best
    )

    SHORT_POOL = 20

    def _segment(self, segment, used, seed_parts):
        """A pseudoword for one letters-only segment; (pseudoword or None, diagnostics)."""
        p = self.profile
        key = self.lexicon.key(segment)
        tu = units(key, p.unit)
        n = self.lexicon.length(segment)                # visible length (ß = 1), the length to match
        seed = self._seed(*seed_parts, key)
        pools = {0: self.model.candidates(tu, seed, p.n_samples, p.temperature, length=n)}
        target_lp = pools[0][1]
        diag = {"target_logprob": round(target_lp, 3), "n_candidates": len(pools[0][0]), "n_rejected_real_word": 0}
        steps = []
        for name, med, extra in self.RELAXATIONS:
            if name == "length":
                steps += [(f"length differs by {k}", med, k) for k in range(1, p.len_tolerance + 1)]
            else:
                steps.append((name, med, extra))

        def pool(k):
            """Candidates k units shorter or longer than the target (both, best match first)."""
            if k not in pools:
                merged = []
                for m in {n - k, n + k}:
                    if m >= 1:
                        merged += self.model.candidates(tu, seed, p.n_samples, p.temperature, length=m)[0]
                pools[k] = (sorted(merged, key=lambda c: (abs(c[1] - target_lp), c[0])), target_lp)
            return pools[k]

        for relaxation, min_ed, extra in steps:
            pool(extra)
            min_ed = p.min_edit_distance if min_ed is None else min_ed
            rejected, passing = 0, []
            for cu, lp in pools[extra][0]:
                cand = "".join(cu)
                if cand == key or cand in used or edit_distance(cu, tu) < min_ed:
                    continue
                short = len(cu) <= p.short_word_max_len
                limit = p.max_repeat_short if short else p.max_repeat
                if limit and self.times_used.get(cand, 0) >= limit:
                    continue
                if self.vowels and not self.vowels & set(cu):
                    continue
                if self.lexicon.is_real_word(cand):
                    rejected += 1
                    continue
                if not short and p.min_word_distance > 1 and self.lexicon.near_common_word(cu, self.unit_set):
                    rejected += 1               # e.g. اننتصب: one letter away from انتصب, read as the word
                    continue
                if not short and self.lexicon.affixed_common_word(cand):
                    rejected += 1               # e.g. تكملتك = تكملة + ك
                    continue
                passing.append((cu, lp))
                # Long words: the best match. Short words: the least-used of the SHORT_POOL best
                # matches, so that "der" does not get the same pseudoword in every sentence.
                if not short or len(passing) >= self.SHORT_POOL:
                    break
            if passing:
                cu, lp = min(passing, key=lambda c: self.times_used.get("".join(c[0]), 0))
                diag.update(distractor_logprob=round(lp, 3), edit_distance=edit_distance(cu, tu),
                            relaxation=relaxation, n_rejected_real_word=rejected)
                return "".join(cu), diag
            diag["n_rejected_real_word"] = rejected
        return None, diag

    def pseudoword(self, body, used, seed_parts=()):
        """A pseudoword for a word body (no edge punctuation), in the target's case.

        Returns (pseudoword or None, report fields)."""
        p = self.profile
        parts = _SEGMENTS.split(body)
        out, diags = [], []
        for k, part in enumerate(parts):
            if k % 2 == 1 or not part:            # separator, or empty edge
                out.append(part)
                continue
            if any(ch.isdigit() for ch in part):
                return None, {"note": "contains digits"}
            if len(units(part, p.unit)) == 1 and len(parts) > 1:
                out.append(part)                  # one letter of a compound (U-Boot): kept as it is
                continue
            pw, diag = self._segment(part, used, (*seed_parts, k))
            if pw is None:
                n = self.lexicon.length(part)
                return None, {**diag, "note": f"no pseudoword of {n}" + (f"-{n + p.len_tolerance}" if p.len_tolerance else "") + " units"}
            out.append(copy_case(part, pw, p.has_case))
            diags.append(diag)
        word = "".join(out)
        if not diags:
            return None, {"note": "no letters"}
        rep = {
            "target_logprob": round(sum(d["target_logprob"] for d in diags) / len(diags), 3),
            "distractor_logprob": round(sum(d["distractor_logprob"] for d in diags) / len(diags), 3),
            "edit_distance": sum(d["edit_distance"] for d in diags),
            "n_candidates": sum(d["n_candidates"] for d in diags),
            "n_rejected_real_word": sum(d["n_rejected_real_word"] for d in diags),
            "relaxation": "; ".join(d["relaxation"] for d in diags if d["relaxation"]),
        }
        return word, rep

    def run(self, items):
        """Fill in `distractors` for every sentence; return the report rows."""
        p = self.profile
        rows = []
        self.times_used = {}                    # repeats are counted per run
        for item in items.values():
            chosen = {}                         # label -> distractor body (no punctuation)
            targets = {}                        # label -> target body, to catch inconsistent labels
            used = set()
            for s in item.sentences:
                for i, (tok, lab) in enumerate(zip(s.words, s.labels)):
                    body = strip_punct(tok)
                    if lab in targets and targets[lab] != body:
                        logging.warning("Item %s: label %s is %r in one row and %r in another; "
                                        "they share one distractor", item.id, lab, targets[lab], body)
                    if (i == 0 and p.first_word) or lab in chosen:
                        targets.setdefault(lab, body)
                        continue
                    targets[lab] = body
                    pw, rep = self.pseudoword(body, used, (item.id, lab))
                    tn = self.lexicon.length(body)
                    if pw is None:
                        logging.warning("Item %s, word %r: %s; placeholder used", item.id, body, rep.get("note"))
                        chosen[lab] = PLACEHOLDER
                    else:
                        chosen[lab] = pw
                        used.add(self.lexicon.key(pw))
                        for seg in _SEGMENTS.split(self.lexicon.key(pw))[::2]:
                            if seg:
                                self.times_used[seg] = self.times_used.get(seg, 0) + 1
                    dn = len(units(self.lexicon.key(pw), p.unit)) if pw else None
                    lp_t, lp_d = rep.get("target_logprob"), rep.get("distractor_logprob")
                    rows.append({
                        "item_id": item.id, "label": lab, "position": i, "target": body,
                        "distractor": chosen[lab], "generator": "neural" if pw else "placeholder",
                        "target_units": tn, "distractor_units": dn, "len_ok": dn == tn if pw else False,
                        "is_real_word": self.lexicon.is_real_word(pw) if pw else None,
                        "edit_distance": rep.get("edit_distance"),
                        "target_logprob": lp_t, "distractor_logprob": lp_d,
                        "logprob_delta": None if lp_t is None or lp_d is None else round(lp_d - lp_t, 3),
                        "n_candidates": rep.get("n_candidates"),
                        "n_rejected_real_word": rep.get("n_rejected_real_word"),
                        "relaxation": rep.get("relaxation", ""),
                        "is_placeholder": pw is None, "note": rep.get("note", ""),
                    })
            for s in item.sentences:
                s.distractors = [copy_punct(tok, PLACEHOLDER if i == 0 and p.first_word else chosen[lab])
                                 for i, (tok, lab) in enumerate(zip(s.words, s.labels))]
        return rows
