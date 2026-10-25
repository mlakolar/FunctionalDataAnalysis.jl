#!/usr/bin/env python3
"""Create or validate complete route/replicate work lists for GNU Parallel."""
import argparse
import csv
from pathlib import Path

from configs import FIELDS


ROUTES = {"dense": ("dense", "face"), "sparse": ("pace", "face_sparse")}
UNIT_FIELDS = ("cell", "route", "first_replicate", "last_replicate", "output")


def positive_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a positive integer, got {value!r}") from error
    if number < 1 or str(number) != str(value):
        raise ValueError(f"expected a positive integer, got {value!r}")
    return number


def read_configurations(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("unexpected scaling configuration CSV header")
        configs = list(reader)
    if not configs:
        raise ValueError("configuration grid is empty")
    for index, config in enumerate(configs, 1):
        if None in config or None in config.values():
            raise ValueError(f"malformed configuration row {index}")
        if positive_integer(config["id"]) != index:
            raise ValueError("configuration id must match its one-based row index")
        if config["design"] not in ROUTES:
            raise ValueError(f"unrecognized configuration design {config['design']!r}")
        positive_integer(config["replicates"])
    return configs


def output_name(cell, route, first, last):
    return f"cell-{cell:03d}-{route}-{first:04d}-{last:04d}.csv"


def overrides(values, configs, per_cell=False):
    """Parse explicit scheduling overrides; reject unused or duplicate keys."""
    result = {}
    allowed = {route for routes in ROUTES.values() for route in routes}
    for value in values:
        try:
            key, raw_size = value.split("=")
            size = positive_integer(raw_size)
            if per_cell:
                raw_cell, route = key.split(":")
                cell = positive_integer(raw_cell)
                if cell > len(configs) or route not in ROUTES[configs[cell-1]["design"]]:
                    raise ValueError("cell or route does not match the configuration")
                key = (cell, route)
            elif key not in allowed or not any(key in ROUTES[c["design"]] for c in configs):
                raise ValueError("route is absent from the configuration")
            if key in result:
                raise ValueError("duplicate override")
        except ValueError as error:
            raise ValueError(f"invalid block-size override {value!r}: {error}") from error
        result[key] = size
    return result


def work_units(configs, block_size, route_sizes=None, cell_sizes=None):
    """Yield each configured (cell, route, replicate) exactly once in order."""
    block_size = positive_integer(block_size)
    route_sizes, cell_sizes = route_sizes or {}, cell_sizes or {}
    for config in configs:
        cell, replicates = int(config["id"]), int(config["replicates"])
        for route in ROUTES[config["design"]]:
            size = positive_integer(cell_sizes.get((cell, route), route_sizes.get(route, block_size)))
            for first in range(1, replicates+1, size):
                last = min(first+size-1, replicates)
                yield dict(zip(UNIT_FIELDS, (cell, route, first, last,
                                            output_name(cell, route, first, last))))


def check_work_units(path, configs):
    """Validate a work list before launch without reading or changing results."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != list(UNIT_FIELDS):
            raise ValueError("unexpected work-list TSV header")
        units = list(reader)
    observed = {}
    for unit in units:
        if None in unit or None in unit.values():
            raise ValueError("malformed work-list row")
        cell = positive_integer(unit["cell"])
        if cell > len(configs):
            raise ValueError(f"work-list cell {cell} is outside the configuration grid")
        config, route = configs[cell-1], unit["route"]
        first, last = (positive_integer(unit[key]) for key in ("first_replicate", "last_replicate"))
        if route not in ROUTES[config["design"]] or not 1 <= first <= last <= int(config["replicates"]):
            raise ValueError(f"work-list route or replicate range does not match cell {cell}")
        if unit["output"] != output_name(cell, route, first, last):
            raise ValueError("work-list output must be the canonical chunk basename")
        seen = observed.setdefault((cell, route), set())
        attempted = set(range(first, last+1))
        if seen & attempted:
            raise ValueError(f"duplicate work-list replicates for cell {cell}, route {route}")
        seen.update(attempted)
    for config in configs:
        cell, replicates = int(config["id"]), int(config["replicates"])
        for route in ROUTES[config["design"]]:
            if observed.get((cell, route), set()) != set(range(1, replicates+1)):
                raise ValueError(f"missing work-list replicates for cell {cell}, route {route}")
    return len(units)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, help="new TSV work list, or existing list with --check")
    parser.add_argument("--block-size", type=int, default=10,
                        help="replicates per worker invocation (default: 10)")
    parser.add_argument("--route-block-size", action="append", default=[], metavar="ROUTE=COUNT")
    parser.add_argument("--cell-block-size", action="append", default=[], metavar="CELL:ROUTE=COUNT")
    parser.add_argument("--check", action="store_true", help="validate full coverage of an existing list")
    args = parser.parse_args()
    configs = read_configurations(args.config)
    if args.check:
        print(f"Validated {check_work_units(args.output, configs)} work units")
        return
    route_sizes = overrides(args.route_block_size, configs)
    cell_sizes = overrides(args.cell_block_size, configs, per_cell=True)
    units = list(work_units(configs, args.block_size, route_sizes, cell_sizes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=UNIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(units)
    print(f"Wrote {len(units)} work units to {args.output}")


if __name__ == "__main__":
    main()
