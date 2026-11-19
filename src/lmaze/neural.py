"""The letter model: a small character-level LSTM, trained per language on the wordfreq word list.

- Training: the training words of the language (lexicon.py), each word once, written as
  <start> unit unit ... unit <end>. The network learns to predict every next unit from all units
  before it in the word. 5 % of the words are held out; training stops when the held-out loss stops
  improving, before the network starts memorising words.
- Generating: candidates of exactly n units are drawn unit by unit from the network's predictions
  (the end marker is blocked until n units are reached), many candidates in one batch.
- Scoring: mean log2 probability per step of a string, the end marker included (= minus the mean
  surprisal per unit). Used to pick the candidate that is about as word-like as its target.

Weights are trained once per language and settings and kept in a cache folder (`cache_dir`).
Training uses a fixed seed; other machines or PyTorch versions can still give slightly different
weights, so collaborators who must get identical pseudowords should share the cached file.
"""

import hashlib
import json
import logging
import math
import random
import time
import unicodedata

import torch
from torch import nn

from .cache import cache_dir

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
ARCH = {"emb": 64, "hidden": 256, "dropout": 0.25}
TRAIN = {"batch": 256, "lr": 3e-3, "max_epochs": 20, "patience": 2, "heldout": 0.05, "seed": 1, "threads": 4}
VERSION = 2                                                # bump when the architecture or training changes
SHORT_MAX = 4                                              # lengths up to this get more samples
SHORT_SAMPLES = 3000


def _base(unit):
    """A unit without its accents (é -> e): how a letter the model never saw is scored, e.g. the é of
    a loanword (Café) in German. Without this it would be scored as <unk>, as if the word were
    extremely unusual."""
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", unit)
                                                 if unicodedata.category(c) != "Mn"))


class _LSTM(nn.Module):
    def __init__(self, vocab, arch):
        super().__init__()
        self.emb = nn.Embedding(vocab, arch["emb"], padding_idx=PAD)
        self.rnn = nn.LSTM(arch["emb"], arch["hidden"], batch_first=True)
        self.drop = nn.Dropout(arch["dropout"])
        self.out = nn.Linear(arch["hidden"], vocab)

    def forward(self, x, state=None):
        h, state = self.rnn(self.drop(self.emb(x)), state)
        return self.out(self.drop(h)), state


class NeuralModel:
    def __init__(self, words, key_parts=(), arch=None, train=None, cache=True, final_only=(), not_final=()):
        self.arch = {**ARCH, **(arch or {})}
        self.train_cfg = {**TRAIN, **(train or {})}
        self.itos = SPECIALS + sorted({u for w in words for u in w})
        self.stoi = {u: i for i, u in enumerate(self.itos)}
        self.net = _LSTM(len(self.itos), self.arch)
        key = hashlib.sha256(json.dumps(
            [VERSION, self.arch, {k: v for k, v in self.train_cfg.items() if k != "threads"}, list(key_parts),
             len(words), ["".join(w) for w in words[:50]], ["".join(w) for w in words[-50:]]],
            ensure_ascii=False, default=str).encode()).hexdigest()[:12]
        lang = key_parts[0] if key_parts else "xx"
        self.path = cache_dir() / f"lmaze-{lang}-{key}.pt"
        if cache and self.path.exists():
            saved = torch.load(self.path, map_location="cpu", weights_only=False)
            self.net.load_state_dict(saved["state"])
            self.info = saved["info"]
        else:
            self.info = self._train(words)
            if cache:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state": self.net.state_dict(), "info": self.info, "itos": self.itos}, self.path)
        self.net.eval()
        self.final_only = [self.stoi[u] for u in final_only if u in self.stoi]
        self.not_final = [self.stoi[u] for u in not_final if u in self.stoi]

    # ---- training ---------------------------------------------------------------------------------
    def _encode(self, seq):
        return [self.stoi[u] if u in self.stoi else self.stoi.get(_base(u), UNK) for u in seq]

    def _batches(self, words, rng=None):
        idx = list(range(len(words)))
        if rng is not None:
            rng.shuffle(idx)
        b = self.train_cfg["batch"]
        for i in range(0, len(idx), b):
            seqs = [[BOS] + self._encode(words[j]) + [EOS] for j in idx[i:i + b]]
            m = max(map(len, seqs))
            x = torch.tensor([s + [PAD] * (m - len(s)) for s in seqs])
            yield x[:, :-1], x[:, 1:]

    @torch.no_grad()
    def bits_per_unit(self, words):
        """Mean surprisal per step in bits (end marker included) over a word list."""
        self.net.eval()
        tot, n = 0.0, 0
        for x, y in self._batches(words):
            logits, _ = self.net(x)
            tot += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                               ignore_index=PAD, reduction="sum").item()
            n += (y != PAD).sum().item()
        return tot / n / math.log(2)

    def _train(self, words):
        cfg = self.train_cfg
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg["threads"])
        torch.manual_seed(cfg["seed"])
        rng = random.Random(cfg["seed"])
        order = list(range(len(words)))
        rng.shuffle(order)
        n_held = max(1, int(len(words) * cfg["heldout"]))
        held = [words[i] for i in order[:n_held]]
        train = [words[i] for i in order[n_held:]]
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        best, best_state, bad, t0, epoch = float("inf"), None, 0, time.time(), 0
        logging.warning("Training the letter model on %d words (once for these settings; it is then "
                        "loaded from %s)", len(words), cache_dir())
        for epoch in range(1, cfg["max_epochs"] + 1):
            self.net.train()
            for x, y in self._batches(train, rng):
                logits, _ = self.net(x)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                                   ignore_index=PAD)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            h = self.bits_per_unit(held)
            logging.info("epoch %d: held-out %.3f bits per unit (%.0f s)", epoch, h, time.time() - t0)
            if h < best - 1e-3:
                best, bad = h, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        info = {"words": len(words), "epochs": epoch, "heldout_bits": round(best, 3),
                "train_bits": round(self.bits_per_unit(train[:4 * len(held)]), 3),
                "seconds": round(time.time() - t0, 1), "torch": torch.__version__}
        torch.set_num_threads(threads)
        return info

    # ---- scoring ----------------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, seqs, batch=4096):
        """Mean log2 probability per step (units and the end marker) for each sequence of units."""
        out = []
        for i in range(0, len(seqs), batch):
            enc = [[BOS] + self._encode(s) + [EOS] for s in seqs[i:i + batch]]
            m = max(map(len, enc))
            x = torch.tensor([e + [PAD] * (m - len(e)) for e in enc])
            logits, _ = self.net(x[:, :-1])
            lp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = (x[:, 1:] != PAD).float()
            out += ((lp * mask).sum(1) / mask.sum(1) / math.log(2)).tolist()
        return out

    # ---- generation -------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n, count, seed, temperature=1.0):
        """`count` sequences of exactly n units, drawn in one batch, one unit position at a time.

        temperature (the usual meaning): each option is weighted by p ** (1 / temperature). 1 follows
        the model; above 1 gives rarer units more chance (more varied strings); below 1 favours the
        most likely units."""
        if temperature <= 0:
            raise ValueError(f"temperature must be above 0, got {temperature}")
        if n < 1:
            return []
        g = torch.Generator().manual_seed(seed)
        x = torch.full((count, 1), BOS, dtype=torch.long)
        state, out = None, []
        for i in range(n):
            logits, state = self.net(x, state)
            logp = torch.log_softmax(logits[:, -1], -1)
            logp[:, [PAD, BOS, EOS, UNK]] = -math.inf      # no end before n units, no special symbols
            banned = self.not_final if i == n - 1 else self.final_only   # spelling rules of the script
            if banned:
                logp[:, banned] = -math.inf
            nxt = torch.multinomial(torch.softmax(logp / temperature, -1), 1, generator=g)
            out.append(nxt)
            x = nxt
        seqs = torch.cat(out, 1).tolist()
        return [tuple(self.itos[i] for i in s) for s in seqs]

    def candidates(self, target_units, seed, n_samples=400, temperature=1.0, length=None):
        """Distinct candidates of the target's length (or `length`), best match first:
        ([(units, score), ...] sorted by |score - target's score|, target's score)."""
        n = len(target_units) if length is None else length
        target_lp = self.score([tuple(target_units)])[0]
        count = max(n_samples, SHORT_SAMPLES) if n <= SHORT_MAX else n_samples
        seqs = list(dict.fromkeys(self.sample(n, count, seed % (2 ** 63), temperature)))
        scored = dict(zip(seqs, self.score(seqs))) if seqs else {}
        return sorted(scored.items(), key=lambda kv: (abs(kv[1] - target_lp), kv[0])), target_lp
