# Automated L-maze

Pseudoword distractors for lexicality-maze (L-maze) experiments: give it your sentences, get back a
matched, orthographically legal pseudoword for every word, ready for PsychoPy, jsPsych or PCIbex,
plus a report of how well each distractor meets the constraints.

Designed so that a new language, in any script, needs only a params file. Included: German,
English, Arabic, Hebrew and Chinese.

**Status:** in development.

## Install (macOS, Linux, Windows)

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/mohamedsaid2710/Automated-L-maze.git
cd Automated-L-maze
uv sync
```

No GPU and no model downloads: the word lists come with the `wordfreq` package.

## Run

```
uv run lmaze -i examples/german_sample.txt -o output_de.txt -p params_de.txt -f delim
```

- `-i` input file, `-o` output file, `-p` params file, `-f` `delim` or `ibex`.
- Input: `tag;id;sentence[;labels]`, UTF-8, one sentence per line. Words with the same label in the
  conditions of one item get the same distractor. Chinese must be segmented into words with spaces.
- Output: `delim` (`tag;id;sentence;distractors;labels`) or `ibex` (PCIbex Maze lines), plus
  `<output>.report.csv` with one row per distractor.

## Parameters

Each language has a params file (`params_de.txt`, `params_en.txt`, `params_ar.txt`, `params_he.txt`,
`params_zh.txt`), in a simple `key: value` format. Every
parameter is explained in the file; edit the values as you need.

Mohammed Said, Institute of Linguistics, University of Stuttgart.
