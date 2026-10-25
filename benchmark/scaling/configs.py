#!/usr/bin/env python3
"""Generate the paired-route long-grid benchmark configuration grid."""
import argparse
import csv
import itertools
from pathlib import Path


FIELDS = ("id", "design", "n", "m", "ratio", "min_points", "max_points",
          "replicates", "seed")


def configurations(smoke=False):
    """Return deterministic cells; both routes in a cell share replicate seeds."""
    if smoke:
        cells = [("dense", 12, 40, 0.05), ("sparse", 12, 21, 0.2)]
    else:
        cells = [("dense", n, m, ratio)
                 for m, n, ratio in itertools.product((100, 500, 2000, 5000),
                                                      (100, 1000), (0.05, 0.2))]
        cells += [("sparse", n, 101, ratio)
                  for n, ratio in itertools.product((100, 1000), (0.05, 0.2))]
    for index, (design, n, m, ratio) in enumerate(cells, 1):
        yield dict(zip(FIELDS, (index, design, n, m, ratio,
                               m if design == "dense" else 2,
                               m if design == "dense" else 10,
                               1 if smoke else 200,
                               20261025 + 1000 * index)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="two small paired cells, one replicate per cell")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(configurations(args.smoke))


if __name__ == "__main__":
    main()
