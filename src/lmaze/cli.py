"""Generate pseudoword distractors for an L-maze experiment.

    lmaze -i german_sample.txt -o output_de.txt -p params_de.txt -f delim

Arguments: input, output, params file, format (delim or ibex).
"""

import argparse
import logging
import sys
import time

from .generate import Generator
from .items import read_input
from .output import save_delim, save_ibex, save_report
from .profile import load_profile


def _summary(rows):
    n = len(rows)
    if not n:
        return "no positions to fill"
    k = lambda f: sum(1 for r in rows if f(r))
    return (f"{n} distractors: length matched {k(lambda r: r['len_ok'])}/{n}, "
            f"real words {k(lambda r: r['is_real_word'])}/{n}, relaxed {k(lambda r: r['relaxation'])}/{n}, "
            f"placeholders {k(lambda r: r['is_placeholder'])}/{n}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lmaze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file", nargs="?", help="input file (or -i)")
    ap.add_argument("output_file", nargs="?", help="output file (or -o)")
    ap.add_argument("params_file", nargs="?", help="params file (or -p)")
    ap.add_argument("-i", "--input", help="stimulus file: tag;id;sentence[;labels]")
    ap.add_argument("-o", "--output", help="output file")
    ap.add_argument("-p", "--params", help="params file, e.g. params_de.txt")
    ap.add_argument("-f", "--format", choices=["delim", "ibex"], default="delim")
    ap.add_argument("--seed", type=int, help="overrides the seed in the params file")
    args = ap.parse_args(argv)
    infile = args.input or args.input_file
    outfile = args.output or args.output_file
    params = args.params or args.params_file
    if not (infile and outfile and params):
        ap.error("input, output and params file are required (-i, -o, -p)")
    # Windows consoles may not be UTF-8: print Hebrew, Arabic, Chinese ... without crashing
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    profile = load_profile(params)
    items = read_input(infile, profile.normalise)
    t = time.time()
    gen = Generator(profile, seed=profile.seed if args.seed is None else args.seed)
    rows = gen.run(items)
    (save_ibex if args.format == "ibex" else save_delim)(outfile, items)
    print(f"{profile.name}: {len(items)} items -> {outfile} ({time.time() - t:.1f} s)")
    print(_summary(rows))
    if profile.write_report:
        print(f"report: {save_report(outfile, rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
