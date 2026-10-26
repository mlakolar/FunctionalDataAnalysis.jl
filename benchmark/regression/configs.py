#!/usr/bin/env python3
"""Generate the paired-route scalar-on-function regression study grid."""
import argparse
import csv
import itertools
from pathlib import Path


FIELDS = ("id", "design", "beta", "n", "r2", "n_test", "replicates", "seed")
DESIGNS = ("dense", "sparse")
BETAS = ("leading", "trailing", "smooth")
ROUTES = ("fpc", "fpc_bic", "pfr")


def configurations(smoke=False):
    """Return deterministic cells; both routes in a cell share replicate seeds."""
    if smoke:
        cells = [("dense", "leading", 40, 0.8), ("sparse", "smooth", 40, 0.5)]
    else:
        cells = list(itertools.product(DESIGNS, BETAS, (100, 500), (0.8, 0.5)))
    for index, (design, beta, n, r2) in enumerate(cells, 1):
        yield dict(zip(FIELDS, (index, design, beta, n, r2,
                               20 if smoke else 200,
                               2 if smoke else 200,
                               20261026 + 1000 * index)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="two small paired cells, two replicates per cell")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(configurations(args.smoke))


if __name__ == "__main__":
    main()
