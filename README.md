# Automated L-maze

Pseudoword distractors for lexicality-maze (L-maze) experiments: give it your sentences, get back a
matched, orthographically legal pseudoword for every word, ready for PsychoPy, jsPsych or PCIbex,
plus a report of how well each distractor meets the constraints.

Designed so that a new language, in any script, needs only a params file. Included: German,
English and Arabic.

**Status:** in development.

## Install (macOS, Linux, Windows)

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/mohamedsaid2710/Automated-L-maze.git
cd Automated-L-maze
uv sync
```

No GPU needed: PyTorch is installed as a CPU build, and the word lists come with the `wordfreq`
package.

## How it works

1. **Training.** For each language, a small neural letter model (a character-level LSTM) is trained
   on up to 100,000 frequent words of the language from `wordfreq` (`train_max_words`). It learns
   which letter can follow the letters before it in a word. 5 % of the words are held out, and
   training stops when the model stops getting better on them, before it memorises words.
2. **Generating.** For every word in your sentences, the model draws a few hundred to a few
   thousand letter strings of the same length.
3. **Filtering.** A string is dropped if it is
   - a real word (any entry of the `wordfreq` list),
   - one letter away from a common word, so that it would read as a misspelling (`min_word_distance`),
   - a word plus prefixes or suffixes (Arabic بالكتاب, تكملتك; `prefixes`, `suffixes`),
   - too close to its target (`min_edit_distance`), or already used too often (`max_repeat`).

   Spelling rules of the script are kept while drawing: Arabic ة ى only at the end of a word
   (`final_only`, `not_final`).
4. **Choosing.** Of the strings left, the tool keeps the one that is about as word-like as its target
   (the model's mean log probability per letter, the negative of mean surprisal). Short words with no
   such string of the same length may get one a letter longer or shorter (`len_tolerance`); this
   is marked in the report.

The first run in a language trains its model (about 5 to 15 minutes on a laptop CPU); after that,
the model is loaded from a cache folder in a second. Set `LMAZE_CACHE` to choose the folder.
Training uses a fixed seed, but other machines or PyTorch versions can give slightly different
models: to get identical pseudowords on several computers, copy the cached `.pt` file.

## Run

```
uv run lmaze -i examples/german_sample.txt -o output_de.txt -p params_de.txt -f delim
```

- `-i` input file, `-o` output file, `-p` params file, `-f` `delim` or `ibex`, `--seed` overrides
  the seed in the params file. Without a seed, each run is different; the seed it used is printed
  and is the first column of the report, so `--seed <that number>` repeats the run exactly.
- Input: `tag;id;sentence[;labels]`, UTF-8, one sentence per line, separated by semicolons, tabs or
  commas (`.txt`, `.csv` or `.tsv`). With a header row, columns are found by name (`tag`, `id` or
  `item`, `sentence`, `labels`) in any order, and other columns are ignored. Words with the same label
  in the conditions of one item get the same distractor.
- Output: `delim` (`tag;id;sentence;distractors;labels`) or `ibex` (PCIbex Maze lines), plus
  `<output>.report.csv` with one row per distractor: target, distractor, lengths, edit distance,
  the word-likeness of both, and any relaxation used. `logprob_delta` is the distractor's minus the
  target's mean log2 probability per letter (negative = the distractor is less word-like);
  `letter_prob_ratio` is the same as a ratio, 2 ** delta: 0.66 means each letter of the distractor is
  on average two thirds as probable as each letter of the target, 1 means equally word-like.

Example (German):

```
sample;2;Die Katze saß auf der Matte.;x-x-x Unehr hlä lym fys Befür.;0 1 2 3 4 5
```

## Parameters

Each language has a params file (`params_de.txt`, `params_en.txt`, `params_ar.txt`) in a simple `key: value` format. Every parameter is explained in the file; edit
the values as you need. The ones you are most likely to change:

| Parameter | What it does |
|---|---|
| `first_word` | `1`: `x-x-x` at the first position (standard Maze); `0`: the first word gets a pseudoword too |
| `seed` | `None` (default): a new random seed each run, printed and written to the report; a number: the same distractors every run |
| `min_edit_distance` | how many letters a distractor must differ from its target |
| `max_repeat`, `max_repeat_short` | how often one distractor may be used in a run (short ones may recur) |
| `len_tolerance` | how much longer or shorter a short word's distractor may be (0 = never) |
| `n_samples` | strings drawn per word; more = closer matches, slower |
| `temperature` | 1 follows the model; above 1 gives rarer letters more chance, below 1 fewer |

A new language needs only its own params file (language code, script, alphabet, spelling rules).

## Tuning the letter model

The defaults work for German, English and Arabic; change them only if you have a reason. There are
two places to change things.

**Training words: the params file.**

| Parameter | What it does |
|---|---|
| `train_min_zipf` | only words at least this frequent are learned (Zipf scale; 3 = once per million words). Lower = more and rarer words, but more web noise (names, typos, foreign words) |
| `train_max_words` | at most this many words, the most frequent first (default 100,000) |
| `alphabet`, `vowels` | words with other letters, or without a vowel (acronyms), are left out |

**The network and its training: `ARCH` and `TRAIN` at the top of `src/lmaze/neural.py`.**

| Setting | Default | What it does |
|---|---|---|
| `emb` | 64 | numbers that represent each letter |
| `hidden` | 256 | size of the model's memory; more = more capacity, but slower and more prone to memorising words |
| `dropout` | 0.25 | share of the network switched off at random during training, against memorising |
| `lr` | 0.003 | learning rate: how big each training step is |
| `max_epochs` | 20 | most passes through the training words |
| `patience` | 2 | stop after this many passes without improvement on the held-out words |
| `heldout` | 0.05 | share of the words kept out of training to check the model |
| `batch` | 256 | words per training step |
| `seed` | 1 | starting point of training (weights, split, order); not the generation `seed` of the params file |

**What happens after a change.** Nothing needs to be deleted: any change to these settings makes the
next run train a new model (a few minutes), which is then cached. The old model stays in the cache
folder and is used again if you go back. A new model gives new distractors, even with the same seed,
so keep the settings fixed within one study.

**How to judge a change.** Lower held-out surprisal is better: the model then predicts the letters of
words it has not seen more accurately. It is stored with each model and can be printed with:

```
uv run python -c "import torch, glob, os; [print(os.path.basename(f), torch.load(f, weights_only=False)['info']) for f in glob.glob(os.path.expanduser('~/.cache/lmaze/*.pt'))]"
```

`heldout_bits` is the mean surprisal per letter on the held-out words, `train_bits` the same on
training words; if `train_bits` falls while `heldout_bits` does not, the model is memorising. On
Windows or macOS, or with `LMAZE_CACHE` set, replace `~/.cache/lmaze` with that cache folder.

For reference, German, measured on a separate test set: learning rate 0.003 gave 2.43 bits per
letter after 20 epochs and 2.40 after 40; 0.001 gave 2.48 and 2.43; 0.01 stopped at 2.49. The
defaults were the best of these; training twice as long gained about 1 %.

Mohammed Said, Institute of Linguistics, University of Stuttgart.

## License

MIT; see [LICENSE](LICENSE).
