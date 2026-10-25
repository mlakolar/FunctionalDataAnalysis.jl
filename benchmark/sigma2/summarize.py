#!/usr/bin/env python3
"""Summarize one-row configuration results using Python's standard library."""
import argparse
import csv
import math
from pathlib import Path

ESTIMATORS = ("surface_middle_half", "surface_full", "rotated_middle_half",
              "rotated_full", "projected_fgm")


def summarize(configs, directory):
    rows = []
    for config in configs:
        path = directory / f"cell-{int(config['id']):03d}.csv"
        with path.open(newline="") as stream:
            result_rows = list(csv.DictReader(stream))
        if len(result_rows) != 1:
            raise ValueError(f"{path}: expected exactly one result row")
        result = result_rows[0]
        if any(result[key] != value for key, value in config.items()):
            raise ValueError(f"{path}: configuration differs from the requested grid")
        truth = float(result["sigma2"])
        if not math.isclose(truth, float(config["ratio"])*1.75, rel_tol=1e-14):
            raise ValueError(f"{path}: incorrect known sigma2")
        replicates = int(config["replicates"])
        for estimator in ESTIMATORS:
            count = int(result[estimator + "_successes"])
            truncated = int(result[estimator + "_truncated"])
            if not 0 <= truncated <= count <= replicates:
                raise ValueError(f"{path}: invalid success/truncation counts")
            mean = float(result[estimator + "_mean"])
            m2 = float(result[estimator + "_m2"])
            squared_errors = float(result[estimator + "_squared_error_sum"])
            if count and not all(math.isfinite(v) for v in (mean, m2, squared_errors)):
                raise ValueError(f"{path}: nonfinite moments of successful estimates")
            rows.append(dict(config, estimator=estimator, sigma2=truth,
                             successes=count, failures=replicates-count, truncated=truncated,
                             bias=mean-truth if count else math.nan,
                             variance=m2/(count-1) if count > 1 else math.nan,
                             mse=squared_errors/count if count else math.nan,
                             julia_version=result["julia_version"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configurations", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    with args.configurations.open(newline="") as stream:
        configs = list(csv.DictReader(stream))
    if not configs or len({c["id"] for c in configs}) != len(configs):
        raise ValueError("configuration grid must be nonempty with unique ids")
    rows = summarize(configs, args.results)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(args.output_prefix) + ".csv")
    md_path = Path(str(args.output_prefix) + ".md")
    if csv_path.exists() or md_path.exists():
        raise FileExistsError("summary outputs already exist; use a new prefix")
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("x") as stream:
        stream.write("# Noise variance benchmark\n\n")
        stream.write("Bias and MSE use the truncated estimate. Variance uses divisor R-1. "
                     "If fits fail, moments are conditional on success; failures are shown. "
                     "No failed replicate is replaced.\n\n")
        stream.write("| Cell | Design | n | Ratio | Weight | Bandwidth | Estimator | "
                     "Successes | Failures | Truncated | Bias | Variance | MSE |\n")
        stream.write("|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            labels = [row[k] for k in ("id", "design", "n", "ratio", "weighting", "bandwidth", "estimator")]
            labels += [f"{row['successes']}/{row['replicates']}", row["failures"], row["truncated"]]
            labels += [f"{row[k]:.8g}" for k in ("bias", "variance", "mse")]
            stream.write("| " + " | ".join(map(str, labels)) + " |\n")
    print(f"Wrote {csv_path} and {md_path}; {len(configs)} configurations, {len(rows)} estimator rows")


if __name__ == "__main__":
    main()
