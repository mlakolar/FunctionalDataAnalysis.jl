#!/usr/bin/env python3
"""Generate 48 paired-data configurations, each evaluating all five estimators."""
import argparse
import csv
import itertools
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="two cells, 20 replicates")
    args = parser.parse_args()
    cells = list(itertools.product(("sparse", "dense"), (100, 400),
                                  (0.05, 0.2, 0.5), ("obs", "subj"), ("default", "cv")))
    if args.smoke:
        cells = [("sparse", 100, 0.05, "obs", "default"),
                 ("dense", 100, 0.2, "subj", "cv")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "design", "n", "ratio", "weighting", "bandwidth", "replicates", "seed"))
        for index, (design, n, ratio, weighting, bandwidth) in enumerate(cells, 1):
            # Same data seeds across bandwidth and weighting choices.
            seed = 20261025 + (design == "dense")*100000 + (n == 400)*10000
            seed += (0.05, 0.2, 0.5).index(ratio)*1000
            writer.writerow((index, design, n, ratio, weighting, bandwidth,
                             20 if args.smoke else 500, seed))


if __name__ == "__main__":
    main()
