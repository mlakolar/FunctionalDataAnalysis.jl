#!/usr/bin/env python3
"""Merge complete whole-cell or chunked regression-study results from retained replicates."""
import argparse
import csv
import math
import statistics
from pathlib import Path

from configs import BETAS, DESIGNS, FIELDS, ROUTES


RESOURCE_METRICS = ("wall_seconds", "allocated_bytes", "process_peak_rss_bytes")
QUALITY_METRICS = ("beta_relative_l2_error", "prediction_relative_error",
                   "pointwise_coverage", "whole_curve_coverage", "band_mean_width",
                   "ncomponents", "term_dof")
METRICS = RESOURCE_METRICS + QUALITY_METRICS
AGGREGATE_METRICS = tuple(m for m in METRICS if m != "process_peak_rss_bytes")
MEDIAN_METRICS = ("beta_relative_l2_error", "prediction_relative_error")
VERSIONS = ("julia_version", "fda_version", "kernel_smoothing_version")
REPLICATE_FIELDS = ("replicate", "seed", "route", "status") + METRICS + (
    "lambda", "band_covariance", "grid_start", "grid_end", "signal_variance", "settings")
FAILURE_FIELDS = ("replicate", "seed", "route", "stage", "error_type", "error")
CHUNK_FIELDS = ("first_replicate", "last_replicate", "routes")
POPULATION_FIELDS = ("signal_variance", "beta_norm", "unidentified_beta_norm")


def read_rows(path, required=()):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: missing or duplicate column names")
        if not set(required).issubset(fields):
            raise ValueError(f"{path}: missing required columns")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def integer(value, label, minimum=0):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected an integer") from error
    if str(number) != str(value) or number < minimum:
        raise ValueError(f"{label}: expected an integer >= {minimum}")
    return number


def finite(value, label, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected a finite number") from error
    if (not math.isfinite(number) or (minimum is not None and number < minimum)
            or (maximum is not None and number > maximum)):
        raise ValueError(f"{label}: invalid numeric value {value!r}")
    return number


def read_configurations(path):
    configs = read_rows(path)
    if not configs:
        raise ValueError(f"{path}: configuration grid is empty")
    for index, config in enumerate(configs, 1):
        if list(config) != list(FIELDS):
            raise ValueError(f"{path}: unexpected configuration columns")
        if integer(config["id"], f"{path}: id", 1) != index:
            raise ValueError(f"{path}: configuration id must match its row index")
        if config["design"] not in DESIGNS or config["beta"] not in BETAS:
            raise ValueError(f"{path}: unrecognized design or beta in row {index}")
        integer(config["n"], f"{path}: n", 20)
        integer(config["n_test"], f"{path}: n_test", 1)
        integer(config["replicates"], f"{path}: replicates", 1)
        integer(config["seed"], f"{path}: seed")
        finite(config["r2"], f"{path}: r2", 0, 1)
    return configs


def moments(values):
    return (statistics.fmean(values) if values else math.nan,
            statistics.stdev(values) if len(values) > 1 else math.nan)


def validate_aggregates(result, path, route, successes, failures, metrics):
    for name, expected in (("successes", successes), ("failures", failures)):
        key = f"{route}_{name}"
        if integer(result.get(key), f"{path}: {key}") != expected:
            raise ValueError(f"{path}: {key} differs from retained records")
    key = f"{route}_process_peak_rss_bytes"
    if finite(result.get(key), f"{path}: {key}", 0) != max(metrics["process_peak_rss_bytes"]):
        raise ValueError(f"{path}: {key} differs from retained records")
    for metric in AGGREGATE_METRICS:
        values = metrics[metric]
        key = f"{route}_{metric}_count"
        if integer(result.get(key), f"{path}: {key}") != len(values):
            raise ValueError(f"{path}: {key} differs from retained records")
        key = f"{route}_{metric}_mean"
        if not values:
            if result.get(key) != "":
                raise ValueError(f"{path}: {key} must be blank when its count is zero")
        elif not math.isclose(finite(result.get(key), f"{path}: {key}"),
                              statistics.fmean(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{path}: {key} differs from retained records")


def chunk_scope(result, path, config):
    if set(CHUNK_FIELDS) - set(result):
        raise ValueError(f"{path}: incomplete chunk metadata")
    first = integer(result["first_replicate"], f"{path}: first_replicate", 1)
    last = integer(result["last_replicate"], f"{path}: last_replicate", 1)
    if last < first or last > int(config["replicates"]):
        raise ValueError(f"{path}: replicate range is outside the configuration")
    routes = tuple(result["routes"].split(";"))
    if not routes or len(set(routes)) != len(routes) or not set(routes) <= set(ROUTES):
        raise ValueError(f"{path}: invalid chunk routes {result['routes']!r}")
    for key in result:
        # Longest matching prefix: "fpc_bic_..." belongs to fpc_bic, not fpc.
        owners = [route for route in ROUTES if key.startswith(f"{route}_")]
        owner = max(owners, key=len) if owners else None
        if owner is not None and owner not in routes:
            raise ValueError(f"{path}: aggregates for unselected route {owner}")
    return first, last, routes


def result_chunks(configs, directory):
    """Discover finalized result rows, grouping files by their declared cell id."""
    grouped = {int(config["id"]): [] for config in configs}
    for path in sorted(directory.glob("cell-*.csv")):
        if path.name.endswith((".replicates.csv", ".failures.csv")):
            continue
        rows = read_rows(path, FIELDS + CHUNK_FIELDS + POPULATION_FIELDS
                         + ("elapsed_seconds",) + VERSIONS)
        if len(rows) != 1:
            raise ValueError(f"{path}: expected exactly one result row")
        identifier = integer(rows[0]["id"], f"{path}: id", 1)
        if identifier not in grouped:
            raise ValueError(f"{path}: unexpected configuration id {identifier}")
        grouped[identifier].append((path, rows[0]))
    return grouped


def read_chunk(path, result, config, paired):
    """Validate one chunk and its diagnostics before admitting its observations."""
    first, last, routes = chunk_scope(result, path, config)
    sidecar = path.with_suffix(".replicates.csv")
    expected = {(r, route) for r in range(first, last + 1) for route in routes}
    seen, failed = set(), set()
    metrics = {route: {key: [] for key in METRICS} for route in routes}
    successes = dict.fromkeys(routes, 0)
    failures = dict.fromkeys(routes, 0)
    signal = finite(result["signal_variance"], f"{path}: signal_variance", 0)
    for record in read_rows(sidecar, REPLICATE_FIELDS):
        replicate = integer(record["replicate"], f"{sidecar}: replicate", 1)
        route = record["route"]
        key = (replicate, route)
        if key not in expected or key in seen:
            raise ValueError(f"{sidecar}: duplicate or unexpected replicate/route {key}")
        seen.add(key)
        if integer(record["seed"], f"{sidecar}: seed") != int(config["seed"]) + replicate - 1:
            raise ValueError(f"{sidecar}: incorrect seed for replicate {replicate}")
        start = finite(record["grid_start"], f"{sidecar}: grid_start")
        end = finite(record["grid_end"], f"{sidecar}: grid_end")
        if not 0 <= start < end <= 1:
            raise ValueError(f"{sidecar}: invalid scoring grid")
        if not math.isclose(finite(record["signal_variance"], f"{sidecar}: signal_variance"),
                            signal, rel_tol=1e-12):
            raise ValueError(f"{sidecar}: signal variance differs from the result row")
        if paired.setdefault(replicate, (start, end)) != (start, end):
            raise ValueError(f"{sidecar}: paired routes have different data for replicate {replicate}")
        status = record["status"]
        if status not in ("success", "failure"):
            raise ValueError(f"{sidecar}: unrecognized fit status {status!r}")
        if status == "failure":
            failures[route] += 1
            failed.add(key)
            if any(record[m] for m in QUALITY_METRICS) or record["settings"]:
                raise ValueError(f"{sidecar}: failed fit has quality metrics or settings")
        else:
            successes[route] += 1
            if not record["settings"] or not record["band_covariance"]:
                raise ValueError(f"{sidecar}: successful fit has no recorded settings")
            if (record["ncomponents"] == "") != (route == "pfr") or \
                    (record["lambda"] == "") != (route != "pfr"):
                raise ValueError(f"{sidecar}: complexity fields do not match route {route}")
        for metric in METRICS:
            value = record[metric]
            if status == "failure" and metric in QUALITY_METRICS:
                continue
            if metric == "ncomponents" and route == "pfr":
                continue
            maximum = 1 if metric in ("pointwise_coverage", "whole_curve_coverage") else None
            number = finite(value, f"{sidecar}: {metric}", 0, maximum)
            if metric in ("ncomponents", "whole_curve_coverage"):
                integer(value, f"{sidecar}: {metric}")
            metrics[route][metric].append(number)
    if seen != expected:
        raise ValueError(f"{sidecar}: missing {len(expected - seen)} declared replicate/route records")
    failure_path = path.with_suffix(".failures.csv")
    logged = set()
    for record in read_rows(failure_path, FAILURE_FIELDS):
        key = (integer(record["replicate"], f"{failure_path}: replicate", 1), record["route"])
        if key not in failed or key in logged:
            raise ValueError(f"{failure_path}: duplicate or unexpected failure {key}")
        logged.add(key)
        if integer(record["seed"], f"{failure_path}: seed") != int(config["seed"]) + key[0] - 1:
            raise ValueError(f"{failure_path}: incorrect failure seed")
        if (record["stage"] not in ("fit", "metrics") or not record["error"]
                or not record["error_type"].strip()):
            raise ValueError(f"{failure_path}: failed fit needs its stage, type and error")
    if logged != failed:
        raise ValueError(f"{failure_path}: missing failed-fit diagnostics")
    for route in routes:
        validate_aggregates(result, path, route, successes[route], failures[route], metrics[route])
    return seen, metrics, successes, failures


def summarize(configs, directory):
    summaries, common_versions = [], None
    chunks = result_chunks(configs, directory)
    for config in configs:
        cell_chunks = chunks[int(config["id"])]
        if not cell_chunks:
            raise ValueError(f"{directory}: missing result chunks for cell {config['id']}")
        expected = {(r, route) for r in range(1, int(config["replicates"]) + 1) for route in ROUTES}
        seen, paired, population = set(), {}, None
        metrics = {route: {key: [] for key in METRICS} for route in ROUTES}
        successes = dict.fromkeys(ROUTES, 0)
        failures = dict.fromkeys(ROUTES, 0)
        for path, result in cell_chunks:
            if any(result[key] != value for key, value in config.items()):
                raise ValueError(f"{path}: configuration differs from the requested grid")
            versions = tuple(result[key] for key in VERSIONS)
            if not all(versions):
                raise ValueError(f"{path}: missing software versions")
            if common_versions is None:
                common_versions = versions
            elif versions != common_versions:
                raise ValueError(f"{path}: software versions differ across chunks/cells")
            values = tuple(result[key] for key in POPULATION_FIELDS)
            if population is None:
                population = values
            elif values != population:
                raise ValueError(f"{path}: population quantities differ within the cell")
            finite(result["elapsed_seconds"], f"{path}: elapsed_seconds", 0)
            chunk_seen, chunk_metrics, chunk_successes, chunk_failures = read_chunk(
                path, result, config, paired)
            if seen & chunk_seen:
                raise ValueError(f"{path}: duplicate replicate/route across chunks")
            seen |= chunk_seen
            for route in chunk_metrics:
                successes[route] += chunk_successes[route]
                failures[route] += chunk_failures[route]
                for metric in METRICS:
                    metrics[route][metric].extend(chunk_metrics[route][metric])
        if seen != expected:
            raise ValueError(f"{directory}: cell {config['id']} is missing "
                             f"{len(expected - seen)} configured replicate/route records")
        for route in ROUTES:
            summary = dict(config, route=route, **dict(zip(POPULATION_FIELDS, population)),
                           successes=successes[route], failures=failures[route])
            for metric in METRICS:
                values = metrics[route][metric]
                summary[f"{metric}_count"] = len(values)
                if metric == "process_peak_rss_bytes":
                    summary[f"{metric}_max"] = max(values) if values else math.nan
                    continue
                summary[f"{metric}_mean"], summary[f"{metric}_sd"] = moments(values)
                if metric in MEDIAN_METRICS:
                    summary[f"{metric}_median"] = statistics.median(values) if values else math.nan
            summary.update(zip(VERSIONS, common_versions))
            summaries.append(summary)
    return summaries


def fmt(value):
    return "" if isinstance(value, float) and math.isnan(value) else f"{value:.3g}"


def write_markdown(path, rows):
    cells = {}
    for row in rows:
        cells.setdefault(row["id"], {})[row["route"]] = row
    with path.open("x") as stream:
        stream.write("# Scalar-on-function regression study\n\n")
        stream.write("Every configured replicate is retained for every route; failures are "
                     "counted and never replaced. Quality summaries are conditional on success. "
                     "Errors are relative: integrated squared coefficient error over the scoring "
                     "grid divided by that of the true coefficient (square root), and held-out "
                     "squared prediction error against the noise-free mean response divided by "
                     "the signal variance. Coverage is that of nominal 95% pointwise bands; "
                     "\"whole curve\" is the fraction of replicates in which the pointwise band "
                     "covers the coefficient at every grid point (no simultaneous band is "
                     "offered). Complexity is the retained component count (FPC) or the "
                     "effective degrees of freedom of the functional term (penalized). "
                     "Peak RSS is the cumulative route-worker process peak.\n\n")
        stream.write("## Accuracy and coverage (medians of relative errors)\n\n")
        stream.write("| Cell | Design | Beta | n | R² | Route | OK | Failures | Beta error (median) | "
                     "Beta error (mean) | Prediction error (median) | Pointwise coverage | "
                     "Whole curve | Band width | Complexity |\n")
        stream.write("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for identifier, routes in cells.items():
            for route in ROUTES:
                row = routes[route]
                values = [identifier, row["design"], row["beta"], row["n"], row["r2"], route,
                          f"{row['successes']}/{row['replicates']}", row["failures"],
                          fmt(row["beta_relative_l2_error_median"]),
                          fmt(row["beta_relative_l2_error_mean"]),
                          fmt(row["prediction_relative_error_median"]),
                          fmt(row["pointwise_coverage_mean"]),
                          fmt(row["whole_curve_coverage_mean"]),
                          fmt(row["band_mean_width_mean"]),
                          fmt(row["ncomponents_mean"] if route != "pfr" else row["term_dof_mean"])]
                stream.write("| " + " | ".join(map(str, values)) + " |\n")
        stream.write("\n## Resources (all recorded attempts)\n\n")
        stream.write("| Cell | Design | n | Route | Wall seconds (mean) | Allocated MB (mean) | "
                     "Max process peak RSS MB |\n|---|---|---:|---|---:|---:|---:|\n")
        for identifier, routes in cells.items():
            for route in ROUTES:
                row = routes[route]
                stream.write(f"| {identifier} | {row['design']} | {row['n']} | {route} | "
                             f"{fmt(row['wall_seconds_mean'])} | "
                             f"{fmt(row['allocated_bytes_mean'] / 1e6)} | "
                             f"{fmt(row['process_peak_rss_bytes_max'] / 1e6)} |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configurations", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    configs = read_configurations(args.configurations)
    rows = summarize(configs, args.results)
    csv_path = Path(str(args.output_prefix) + ".csv")
    md_path = Path(str(args.output_prefix) + ".md")
    if csv_path.exists() or md_path.exists():
        raise FileExistsError("summary outputs already exist; use a new prefix")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path} and {md_path}; {len(configs)} configurations, {len(rows)} route rows")


if __name__ == "__main__":
    main()
