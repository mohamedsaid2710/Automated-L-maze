"""Generic unit-recombination generator: works for any script, learned from a word list alone.

1. Every training word is a sequence of units (grapheme clusters, or characters for Chinese),
   padded with start and end markers.
2. The model counts unit n-grams (trigrams by default). For Chinese, whose words are mostly two
   characters long, any n-gram would be the whole word, so order 1 uses positions instead: a
   character that starts words, then characters that occur inside words, then one that ends words
   (PositionalModel), which gives new combinations of real characters. A candidate is built unit by
   unit, and every n-gram in it, including the word start and the word end, must occur in the
   lexicon, so candidates are orthographically legal by construction (e.g., Hebrew final letters
   can only appear word-finally, because that is the only place the lexicon has them).
3. Candidates have exactly the target's number of units: a precomputed table says, for every
   state, which lengths can still be completed, so sampling never runs into a dead end.
4. Candidates are ranked by how close their mean transition log-probability is to the target's,
   so that the distractor is about as word-like as the target (cf. Wuggy).
"""

import math
import random
from collections import defaultdict

BOS, EOS = "\x02", "\x03"


class _Ranking:
    enumerate_max_len = 4          # lengths up to this are enumerated exhaustively
    enumerate_limit = 200_000      # ... unless there are more legal sequences than this

    def _enumerated(self, n):
        """All legal sequences of length n with their log-probabilities (computed once per length)."""
        cache = self.__dict__.setdefault("_enum_cache", {})
        if n not in cache:
            every = self.enumerate(n, limit=self.enumerate_limit)
            cache[n] = None if every is None else {s: self.logprob(s)[0] for s in every}
        return cache[n]

    def candidates(self, target_units, seed, n_samples=400, temperature=0.5, length=None):
        """Distinct legal sequences of the target's length (or `length`), best match first.

        Returns ([(units, mean_logprob), ...] sorted by |mean_logprob - target's|, target's
        mean_logprob). Deterministic for a given seed.
        """
        n = len(target_units) if length is None else length
        rng = random.Random(seed)
        target_lp, _ = self.logprob(target_units)
        seen = {}
        # Short words: list every legal sequence instead of sampling, so that the most word-like
        # unlisted strings are found even when nearly all legal short strings are real words
        if n <= self.enumerate_max_len:
            scored = self._enumerated(n)
            if scored is not None:
                return sorted(scored.items(), key=lambda kv: (abs(kv[1] - target_lp), kv[0])), target_lp
        for _ in range(n_samples):
            s = self.sample(n, rng, temperature)
            if s is None:
                break
            if s not in seen:
                seen[s] = self.logprob(s)[0]
        return sorted(seen.items(), key=lambda kv: (abs(kv[1] - target_lp), kv[0])), target_lp


def make_model(words, order=3, min_count=1, max_len=40):
    if order == 1:
        return PositionalModel(words, min_count=min_count, max_len=max_len)
    return NgramModel(words, order=order, min_count=min_count, max_len=max_len)


class PositionalModel(_Ranking):
    """Order 1: which units start, continue, and end words (and which are one-unit words)."""

    def __init__(self, words, min_count=1, max_len=40):
        pos = {"single": defaultdict(int), "start": defaultdict(int), "mid": defaultdict(int), "end": defaultdict(int)}
        for w in words:
            if len(w) == 1:
                pos["single"][w[0]] += 1
                continue
            pos["start"][w[0]] += 1
            pos["end"][w[-1]] += 1
            for u in w[1:-1]:
                pos["mid"][u] += 1
        self.counts = {k: dict(v) for k, v in pos.items()}
        self.totals = {k: sum(v.values()) for k, v in self.counts.items()}
        self.use = {k: {u: n for u, n in v.items() if n >= min_count} for k, v in self.counts.items()}
        self.vocab_size = len({u for w in words for u in w}) + 1
        self.max_len = max_len

    def _slots(self, n):
        return ["single"] if n == 1 else ["start"] + ["mid"] * (n - 2) + ["end"]

    def can_make(self, n):
        return 0 < n <= self.max_len and all(self.use[s] for s in self._slots(n))

    def sample(self, n, rng, temperature=0.5):
        if not self.can_make(n):
            return None
        out = []
        for slot in self._slots(n):
            opts = self.use[slot]
            out.append(rng.choices(list(opts), weights=[k ** temperature for k in opts.values()])[0])
        return tuple(out)

    def enumerate(self, n, limit=200_000):
        """Every legal sequence of exactly n units, or None if there are more than `limit`."""
        if not self.can_make(n):
            return []
        total = 1
        for slot in self._slots(n):
            total *= len(self.use[slot])
        if total > limit:
            return None
        import itertools
        return list(itertools.product(*(list(self.use[s]) for s in self._slots(n))))

    def logprob(self, seq, k=0.1):
        slots = self._slots(len(seq))
        total, unseen = 0.0, 0
        for u, slot in zip(seq, slots):
            count = self.counts[slot].get(u, 0)
            unseen += count == 0
            total += math.log2((count + k) / (self.totals[slot] + k * self.vocab_size))
        return total / len(seq), unseen


class NgramModel(_Ranking):
    def __init__(self, words, order=3, max_len=40, min_count=1):
        if order < 2:
            raise ValueError("order must be at least 2")
        self.k = order - 1                                  # context length
        grams = defaultdict(lambda: defaultdict(int))
        for w in words:
            seq = (BOS,) * self.k + tuple(w) + (EOS,)
            for i in range(len(seq) - self.k):
                grams[seq[i:i + self.k]][seq[i + self.k]] += 1
        # All counts go into the probabilities (logprob); only n-grams seen at least min_count times
        # may be used to build candidates, which keeps one-off sequences from loanwords and typos out
        self.totals = {ctx: sum(nxt.values()) for ctx, nxt in grams.items()}
        self.counts = {ctx: dict(nxt) for ctx, nxt in grams.items()}
        self.tri = {ctx: kept for ctx, nxt in grams.items()
                    if (kept := {c: n for c, n in nxt.items() if n >= min_count})}
        self.vocab_size = len({u for w in words for u in w}) + 1
        self._reach = [{ctx for ctx, nxt in self.tri.items() if EOS in nxt}]
        self.max_len = max_len
        self.start = (BOS,) * self.k

    def _reachable(self, r):
        """States from which exactly r more units and then the word end are possible."""
        while len(self._reach) <= r:
            prev = self._reach[-1]
            self._reach.append({ctx for ctx, nxt in self.tri.items()
                                if any(c != EOS and ctx[1:] + (c,) in prev for c in nxt)})
        return self._reach[r]

    def can_make(self, n):
        return 0 < n <= self.max_len and self.start in self._reachable(n)

    def sample(self, n, rng, temperature=0.5):
        """One random legal sequence of exactly n units, or None if the lexicon has none."""
        if not self.can_make(n):
            return None
        state, out = self.start, []
        for i in range(n):
            ok = self._reachable(n - i - 1)
            options = [(c, k) for c, k in self.tri[state].items() if c != EOS and state[1:] + (c,) in ok]
            c = rng.choices([c for c, _ in options], weights=[k ** temperature for _, k in options])[0]
            out.append(c)
            state = state[1:] + (c,)
        return tuple(out)

    def enumerate(self, n, limit=200_000):
        """Every legal sequence of exactly n units, or None if there are more than `limit`."""
        if not self.can_make(n):
            return []
        out = []

        def walk(state, prefix):
            r = n - len(prefix)
            if r == 0:
                out.append(tuple(prefix))
                return len(out) <= limit
            ok = self._reachable(r - 1)
            for c in self.tri[state]:
                if c != EOS and state[1:] + (c,) in ok:
                    prefix.append(c)
                    if not walk(state[1:] + (c,), prefix):
                        return False
                    prefix.pop()
            return True

        return out if walk(self.start, []) else None

    def logprob(self, seq, k=0.1):
        """Mean log2 transition probability of a unit sequence (add-k smoothed), and the number of
        unattested n-grams in it."""
        padded = self.start + tuple(seq) + (EOS,)
        total, unseen = 0.0, 0
        steps = len(padded) - self.k
        for i in range(steps):
            ctx, c = padded[i:i + self.k], padded[i + self.k]
            count = self.counts.get(ctx, {}).get(c, 0)
            unseen += count == 0
            total += math.log2((count + k) / (self.totals.get(ctx, 0) + k * self.vocab_size))
        return total / steps, unseen
