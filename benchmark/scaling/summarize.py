#!/usr/bin/env python3
"""Merge complete whole-cell or chunked scaling results from retained replicates."""
import argparse
import csv
import math
import statistics
from pathlib import Path

from configs import FIELDS


ROUTES = {"dense": ("dense", "face"), "sparse": ("pace", "face_sparse")}
RESOURCE_METRICS = ("wall_seconds", "allocated_bytes", "process_peak_rss_bytes")
QUALITY_METRICS = ("eigenvalue_relative_error", "eigenfunction_l2_error",
                   "sigma2_error", "ncomponents", "reconstruction_error")
METRICS = RESOURCE_METRICS + QUALITY_METRICS
AGGREGATE_METRICS = tuple(metric for metric in METRICS
                          if metric != "process_peak_rss_bytes")
VERSIONS = ("julia_version", "fda_version", "kernel_smoothing_version")
REPLICATE_FIELDS = ("replicate", "seed", "route", "status") + METRICS + (
    "grid_start", "grid_end", "truth_eigenvalues", "settings")
FAILURE_FIELDS = ("replicate", "seed", "route", "stage", "error")
CHUNK_FIELDS = ("first_replicate", "last_replicate", "routes")


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


def finite(value, label, minimum=None):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected a finite number") from error
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{label}: invalid numeric value {value!r}")
    return number


def read_configurations(path):
    configs = read_rows(path)
    if not configs:
        raise ValueError(f"{path}: configuration grid is empty")
    identifiers = set()
    for config in configs:
        if set(config) != set(FIELDS):
            raise ValueError(f"{path}: unexpected configuration columns")
        identifier = integer(config["id"], f"{path}: id", 1)
        if identifier in identifiers:
            raise ValueError(f"{path}: duplicate configuration id {identifier}")
        identifiers.add(identifier)
        for key in ("n", "m", "min_points", "max_points"):
            integer(config[key], f"{path}: {key}", 2)
        integer(config["replicates"], f"{path}: replicates", 1)
        integer(config["seed"], f"{path}: seed")
        finite(config["ratio"], f"{path}: ratio", 0)
        if config["design"] not in ROUTES:
            raise ValueError(f"{path}: unrecognized design {config['design']!r}")
        if int(config["max_points"]) < int(config["min_points"]):
            raise ValueError(f"{path}: max_points is smaller than min_points")
        if config["design"] == "dense" and not (
                config["min_points"] == config["m"] == config["max_points"]):
            raise ValueError(f"{path}: dense observation counts must equal m")
    return configs


def moments(values):
    return (statistics.fmean(values) if values else math.nan,
            statistics.stdev(values) if len(values) > 1 else math.nan)


def validate_aggregates(result, path, route, successes, failures, metrics):
    for count_name, expected in (("successes", successes), ("failures", failures)):
        key = f"{route}_{count_name}"
        if integer(result.get(key), f"{path}: {key}") != expected:
            raise ValueError(f"{path}: {key} differs from retained records")
    key = f"{route}_process_peak_rss_bytes"
    peak = finite(result.get(key), f"{path}: {key}", 0)
    if peak != max(metrics["process_peak_rss_bytes"]):
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
        else:
            mean = finite(result.get(key), f"{path}: {key}")
            if not math.isclose(mean, statistics.fmean(values), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{path}: {key} differs from retained records")


def chunk_scope(result, path, config):
    """Return the declared coverage; old result rows cover the entire cell."""
    present = set(CHUNK_FIELDS).intersection(result)
    routes = ROUTES[config["design"]]
    if not present:
        return 1, int(config["replicates"]), routes, False
    if present != set(CHUNK_FIELDS):
        raise ValueError(f"{path}: incomplete chunk metadata")
    first = integer(result["first_replicate"], f"{path}: first_replicate", 1)
    last = integer(result["last_replicate"], f"{path}: last_replicate", 1)
    if last < first or last > int(config["replicates"]):
        raise ValueError(f"{path}: replicate range is outside the configuration")
    selected_routes = tuple(result["routes"].split(";"))
    if (not selected_routes or len(set(selected_routes)) != len(selected_routes)
            or not set(selected_routes).issubset(routes)):
        raise ValueError(f"{path}: invalid chunk routes {result['routes']!r}")
    for route in set(routes) - set(selected_routes):
        if any(key.startswith(f"{route}_") for key in result):
            raise ValueError(f"{path}: aggregates for unselected route {route}")
    return first, last, selected_routes, True


def result_chunks(configs, directory):
    """Discover finalized result rows, grouping files by their declared cell id."""
    grouped = {int(config["id"]): [] for config in configs}
    for path in sorted(directory.glob("cell-*.csv")):
        if path.name.endswith((".replicates.csv", ".failures.csv")):
            continue
        rows = read_rows(path, FIELDS + ("sigma2", "elapsed_seconds") + VERSIONS)
        if len(rows) != 1:
            raise ValueError(f"{path}: expected exactly one result row")
        result = rows[0]
        identifier = integer(result["id"], f"{path}: id", 1)
        if identifier not in grouped:
            raise ValueError(f"{path}: unexpected configuration id {identifier}")
        grouped[identifier].append((path, result))
    return grouped


def read_chunk(path, result, config, paired_metadata):
    """Validate one chunk and its diagnostics before admitting its observations."""
    first, last, routes, explicit_scope = chunk_scope(result, path, config)
    sidecar = path.with_suffix(".replicates.csv")
    records = read_rows(sidecar, REPLICATE_FIELDS)
    expected = {(replicate, route) for replicate in range(first, last + 1)
                for route in routes}
    seen = set()
    metrics = {route: {key: [] for key in METRICS} for route in routes}
    successes = {route: 0 for route in routes}
    failures = {route: 0 for route in routes}
    failed_keys = set()
    for record in records:
        replicate = integer(record.get("replicate"), f"{sidecar}: replicate", 1)
        route = record.get("route")
        key = (replicate, route)
        if key not in expected or key in seen:
            raise ValueError(f"{sidecar}: duplicate or unexpected replicate/route {key}")
        seen.add(key)
        seed = integer(record.get("seed"), f"{sidecar}: seed")
        if seed != int(config["seed"]) + replicate - 1:
            raise ValueError(f"{sidecar}: incorrect seed for replicate {replicate}")
        start = finite(record.get("grid_start"), f"{sidecar}: grid_start")
        end = finite(record.get("grid_end"), f"{sidecar}: grid_end")
        if end <= start:
            raise ValueError(f"{sidecar}: grid_end must exceed grid_start")
        if not record.get("truth_eigenvalues"):
            raise ValueError(f"{sidecar}: missing truth eigenvalues")
        metadata = (start, end, record["truth_eigenvalues"])
        if replicate in paired_metadata and metadata != paired_metadata[replicate]:
            raise ValueError(f"{sidecar}: paired routes have different data metadata "
                             f"for replicate {replicate}")
        paired_metadata[replicate] = metadata
        status = record.get("status")
        if status not in ("success", "failure"):
            raise ValueError(f"{sidecar}: unrecognized fit status {status!r}")
        if status == "failure":
            failures[route] += 1
            failed_keys.add(key)
        else:
            successes[route] += 1
            if not record.get("settings"):
                raise ValueError(f"{sidecar}: successful fit has no recorded settings")
        for metric in METRICS:
            value = record.get(metric)
            if status == "failure" and metric in QUALITY_METRICS:
                if value:
                    raise ValueError(f"{sidecar}: failed fit has a quality metric")
                continue
            if metric == "sigma2_error" and not value and route == "dense":
                continue
            number = finite(value, f"{sidecar}: {metric}",
                            None if metric == "sigma2_error" else 0)
            if metric == "ncomponents":
                integer(value, f"{sidecar}: ncomponents", 1)
            metrics[route][metric].append(number)
    if seen != expected:
        raise ValueError(f"{sidecar}: missing {len(expected - seen)} declared "
                         "replicate/route records")
    failure_path = path.with_suffix(".failures.csv")
    required = FAILURE_FIELDS + (("error_type",) if explicit_scope else ())
    failure_records = read_rows(failure_path, required)
    logged_failures = set()
    for record in failure_records:
        replicate = integer(record.get("replicate"), f"{failure_path}: replicate", 1)
        route = record.get("route")
        key = (replicate, route)
        if key not in failed_keys or key in logged_failures:
            raise ValueError(f"{failure_path}: duplicate or unexpected failure {key}")
        logged_failures.add(key)
        seed = integer(record.get("seed"), f"{failure_path}: seed")
        if seed != int(config["seed"]) + replicate - 1:
            raise ValueError(f"{failure_path}: incorrect failure seed")
        if record.get("stage") not in ("fit", "metrics") or not record.get("error"):
            raise ValueError(f"{failure_path}: failed fit needs its stage and error")
        if explicit_scope and not record.get("error_type", "").strip():
            raise ValueError(f"{failure_path}: failed fit needs its exception type")
    if logged_failures != failed_keys:
        raise ValueError(f"{failure_path}: missing failed-fit diagnostics")
    for route in routes:
        validate_aggregates(result, path, route, successes[route], failures[route],
                            metrics[route])
    return seen, metrics, successes, failures


def summarize(configs, directory):
    summaries = []
    common_versions = None
    chunks = result_chunks(configs, directory)
    for config in configs:
        cell_chunks = chunks[int(config["id"])]
        if not cell_chunks:
            raise ValueError(f"{directory}: missing result chunks for cell {config['id']}")
        replicates = int(config["replicates"])
        routes = ROUTES[config["design"]]
        expected = {(replicate, route) for replicate in range(1, replicates + 1)
                    for route in routes}
        seen = set()
        metrics = {route: {key: [] for key in METRICS} for route in routes}
        successes = {route: 0 for route in routes}
        failures = {route: 0 for route in routes}
        paired_metadata = {}
        for path, result in cell_chunks:
            if any(result.get(key) != value for key, value in config.items()):
                raise ValueError(f"{path}: configuration differs from the requested grid")
            versions = tuple(result.get(key, "") for key in VERSIONS)
            if not all(versions):
                raise ValueError(f"{path}: missing software versions")
            if common_versions is None:
                common_versions = versions
            elif versions != common_versions:
                raise ValueError(f"{path}: software versions differ across chunks/cells")
            truth = finite(result.get("sigma2"), f"{path}: sigma2", 0)
            if not math.isclose(truth, float(config["ratio"]) * 1.75, rel_tol=1e-14):
                raise ValueError(f"{path}: incorrect known sigma2")
            finite(result.get("elapsed_seconds"), f"{path}: elapsed_seconds", 0)
            chunk_seen, chunk_metrics, chunk_successes, chunk_failures = read_chunk(
                path, result, config, paired_metadata)
            if seen.intersection(chunk_seen):
                raise ValueError(f"{path}: duplicate replicate/route across chunks")
            seen.update(chunk_seen)
            for route in chunk_metrics:
                successes[route] += chunk_successes[route]
                failures[route] += chunk_failures[route]
                for metric in METRICS:
                    metrics[route][metric].extend(chunk_metrics[route][metric])
        if seen != expected:
            raise ValueError(f"{directory}: cell {config['id']} is missing "
                             f"{len(expected - seen)} configured replicate/route records")
        for route in routes:
            summary = dict(config, route=route, sigma2=truth,
                           successes=successes[route], failures=failures[route])
            for metric in METRICS:
                values = metrics[route][metric]
                summary[f"{metric}_count"] = len(values)
                if metric == "process_peak_rss_bytes":
                    summary[f"{metric}_max"] = max(values) if values else math.nan
                else:
                    mean, sd = moments(values)
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_sd"] = sd
            summary.update(zip(VERSIONS, common_versions))
            summaries.append(summary)
    return summaries


def write_markdown(path, rows):
    with path.open("x") as stream:
        stream.write("# Long-grid FPCA benchmark\n\n")
        stream.write("Each configured replicate is retained for both routes. Failures are "
                     "counted and are never replaced. Accuracy and component summaries "
                     "are conditional on success. Resource summaries include recorded "
                     "failed attempts; their CSV counts show measurement coverage. "
                     "Peak RSS is the maximum cumulative route-worker process peak, "
                     "including startup, JIT, simulation, fitting, and metrics. "
                     "Dense FPCA has no noise-variance estimate; its sigma2-error count is zero. "
                     "SD uses divisor R-1 and is undefined for one observation.\n\n")
        stream.write("| Cell | Design | n | m | Ratio | Route | Successes | Failures | "
                     "Wall seconds | Allocated bytes | Max process peak RSS bytes | Eigenvalue error | "
                     "Eigenfunction L2 error | Sigma2 error | Components | Reconstruction error |\n")
        stream.write("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            values = [row[key] for key in ("id", "design", "n", "m", "ratio", "route")]
            values += [f"{row['successes']}/{row['replicates']}", row["failures"]]
            for metric in METRICS:
                statistic = "max" if metric == "process_peak_rss_bytes" else "mean"
                values.append(f"{row[f'{metric}_{statistic}']:.8g}")
            stream.write("| " + " | ".join(map(str, values)) + " |\n")


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
    print(f"Wrote {csv_path} and {md_path}; {len(configs)} configurations, "
          f"{len(rows)} route rows")


if __name__ == "__main__":
    main()
