# Automated L-maze

Pseudoword distractors for lexicality-maze (L-maze) experiments: give it your sentences, get back a
matched, orthographically legal pseudoword for every word, ready for PsychoPy, jsPsych or PCIbex,
plus a report of how well each distractor meets the constraints.

Designed so that a new language, in any script, needs only a params file. Included: German,
English, Arabic and Hebrew.

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

   Spelling rules of the script are kept while drawing: Hebrew final letters (ך ם ן ף ץ) and Arabic
   ة ى only at the end of a word (`final_only`, `not_final`).
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
  the seed in the params file.
- Input: `tag;id;sentence[;labels]`, UTF-8, one sentence per line. Words with the same label in the
  conditions of one item get the same distractor.
- Output: `delim` (`tag;id;sentence;distractors;labels`) or `ibex` (PCIbex Maze lines), plus
  `<output>.report.csv` with one row per distractor: target, distractor, lengths, edit distance,
  the word-likeness of both, and any relaxation used.

Example (German):

```
sample;2;Die Katze saß auf der Matte.;x-x-x Unehr hlä lym fys Befür.;0 1 2 3 4 5
```

## Parameters

Each language has a params file (`params_de.txt`, `params_en.txt`, `params_ar.txt`,
`params_he.txt`) in a simple `key: value` format. Every parameter is explained in the file; edit
the values as you need. The ones you are most likely to change:

| Parameter | What it does |
|---|---|
| `first_word` | `1`: `x-x-x` at the first position (standard Maze); `0`: the first word gets a pseudoword too |
| `seed` | same input + same seed = same output |
| `min_edit_distance` | how many letters a distractor must differ from its target |
| `max_repeat`, `max_repeat_short` | how often one distractor may be used in a run (short ones may recur) |
| `len_tolerance` | how much longer or shorter a short word's distractor may be (0 = never) |
| `n_samples` | strings drawn per word; more = closer matches, slower |
| `temperature` | 1 follows the model; below 1 gives rarer letters more chance |

A new language needs only its own params file (language code, script, alphabet, spelling rules).

Mohammed Said, Institute of Linguistics, University of Stuttgart.
