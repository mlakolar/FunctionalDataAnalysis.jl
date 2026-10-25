#!/usr/bin/env python3
"""Focused chunk-merging checks using synthetic records; no Julia fits run."""
import csv
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from configs import FIELDS
from summarize import (AGGREGATE_METRICS, FAILURE_FIELDS, METRICS, QUALITY_METRICS,
                       REPLICATE_FIELDS, ROUTES, VERSIONS, read_configurations,
                       read_rows, summarize)


def write_rows(path, rows, fields=None):
    fields = fields or tuple(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class SummaryChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = dict(zip(FIELDS, map(str, (1, "dense", 12, 40, 0.05,
                                                40, 40, 4, 20262025))))
        self.serial = 0

    def write_chunk(self, first=1, last=4, routes=("dense", "face"), failures=(),
                    legacy=False):
        self.serial += 1
        path = self.directory / f"cell-001-chunk-{self.serial:03d}.csv"
        records = []
        diagnostics = []
        for replicate in range(first, last + 1):
            for route in routes:
                failed = (replicate, route) in failures
                record = dict(replicate=str(replicate),
                              seed=str(int(self.config["seed"]) + replicate - 1),
                              route=route, status="failure" if failed else "success")
                for metric in METRICS:
                    value = 3 if metric == "ncomponents" else replicate
                    record[metric] = ("" if (failed and metric in QUALITY_METRICS)
                                      or (metric == "sigma2_error" and route == "dense")
                                      else str(value))
                record.update(grid_start="0.0", grid_end="1.0",
                              truth_eigenvalues="1.0;0.5;0.25",
                              settings="" if failed else "fixed test settings")
                records.append(record)
                if failed:
                    diagnostic = {key: record[key] for key in ("replicate", "seed", "route")}
                    diagnostic.update(stage="metrics", error="synthetic exception")
                    if not legacy:
                        diagnostic["error_type"] = "MethodError"
                    diagnostics.append(diagnostic)
        result = dict(self.config, sigma2=str(float(self.config["ratio"]) * 1.75),
                      elapsed_seconds="1.0")
        result.update(zip(VERSIONS, ("1.10.10", "0.1.0", "0.1.0")))
        if not legacy:
            result.update(first_replicate=str(first), last_replicate=str(last),
                          routes=";".join(routes))
        for route in routes:
            retained = [record for record in records if record["route"] == route]
            result[f"{route}_successes"] = str(sum(row["status"] == "success" for row in retained))
            result[f"{route}_failures"] = str(sum(row["status"] == "failure" for row in retained))
            result[f"{route}_process_peak_rss_bytes"] = str(max(
                float(row["process_peak_rss_bytes"]) for row in retained))
            for metric in AGGREGATE_METRICS:
                values = [float(row[metric]) for row in retained if row[metric]]
                result[f"{route}_{metric}_count"] = str(len(values))
                result[f"{route}_{metric}_mean"] = str(statistics.fmean(values)) if values else ""
        write_rows(path, [result])
        write_rows(path.with_suffix(".replicates.csv"), records, REPLICATE_FIELDS)
        failure_fields = FAILURE_FIELDS + (() if legacy else ("error_type",))
        write_rows(path.with_suffix(".failures.csv"), diagnostics, failure_fields)
        return path

    def change(self, path, mutation):
        rows = read_rows(path)
        mutation(rows)
        write_rows(path, rows)

    def summaries(self):
        return summarize([self.config], self.directory)

    def assert_invalid(self, text):
        return self.assertRaisesRegex(ValueError, text)

    def test_retained_legacy_smoke(self):
        smoke = Path(__file__).with_name("smoke")
        configs = read_configurations(smoke / "configurations.csv")
        actual = summarize(configs, smoke)
        expected = read_rows(smoke / "summary.csv")
        self.assertEqual(len(actual), len(expected))
        for summary, reference in zip(actual, expected):
            self.assertEqual(set(summary), set(reference))
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    target = float(reference[key])
                    self.assertTrue((math.isnan(value) and math.isnan(target))
                                    or math.isclose(value, target, rel_tol=1e-12), key)
                else:
                    self.assertEqual(value, reference[key], key)

    def test_chunks_merge_to_whole_cell_with_failures(self):
        failures = ((2, "face"), (4, "dense"))
        whole = self.write_chunk(failures=failures)
        expected = self.summaries()
        for suffix in (".csv", ".replicates.csv", ".failures.csv"):
            whole.with_suffix(suffix).unlink()
        self.write_chunk(1, 2, ("dense",), failures)
        self.write_chunk(3, 4, ("dense",), failures)
        self.write_chunk(1, 1, ("face",), failures)
        self.write_chunk(2, 4, ("face",), failures)
        self.assertEqual(self.summaries(), expected)
        dense, face = expected
        self.assertEqual((dense["successes"], dense["failures"]), (3, 1))
        self.assertEqual((face["successes"], face["failures"]), (3, 1))
        self.assertEqual(dense["wall_seconds_count"], 4)
        self.assertEqual(dense["wall_seconds_mean"], 2.5)
        self.assertEqual(dense["process_peak_rss_bytes_max"], 4)
        self.assertEqual(dense["sigma2_error_count"], 0)
        self.assertEqual(face["eigenvalue_relative_error_count"], 3)

    def test_chunked_pair_and_single_routes(self):
        self.write_chunk(1, 2)
        self.write_chunk(3, 4, ("dense",))
        self.write_chunk(3, 4, ("face",))
        self.assertEqual([row["successes"] for row in self.summaries()], [4, 4])

    def test_sparse_routes(self):
        self.config.update(design="sparse", m="21", min_points="2", max_points="10")
        for route in ROUTES["sparse"]:
            self.write_chunk(1, 2, (route,))
            self.write_chunk(3, 4, (route,))
        self.assertEqual([row["route"] for row in self.summaries()], ["pace", "face_sparse"])

    def test_legacy_failure_diagnostics_need_no_type(self):
        self.write_chunk(failures=((1, "dense"),), legacy=True)
        self.assertEqual(self.summaries()[0]["failures"], 1)

    def test_missing_cell(self):
        with self.assert_invalid("missing result chunks"):
            self.summaries()

    def test_missing_chunk(self):
        self.write_chunk(1, 3)
        with self.assert_invalid("missing 2 configured"):
            self.summaries()

    def test_duplicate_across_chunks(self):
        self.write_chunk()
        self.write_chunk(2, 2, ("face",))
        with self.assert_invalid("duplicate replicate/route across chunks"):
            self.summaries()

    def test_unexpected_cell(self):
        path = self.write_chunk()
        self.change(path, lambda rows: rows[0].update(id="2"))
        with self.assert_invalid("unexpected configuration id"):
            self.summaries()

    def test_mismatched_configuration(self):
        path = self.write_chunk()
        self.change(path, lambda rows: rows[0].update(n="13"))
        with self.assert_invalid("configuration differs"):
            self.summaries()

    def test_mismatched_versions(self):
        self.write_chunk(1, 2)
        path = self.write_chunk(3, 4)
        self.change(path, lambda rows: rows[0].update(julia_version="1.13.2"))
        with self.assert_invalid("software versions differ"):
            self.summaries()

    def test_incorrect_seed(self):
        path = self.write_chunk().with_suffix(".replicates.csv")
        self.change(path, lambda rows: rows[0].update(seed="99"))
        with self.assert_invalid("incorrect seed"):
            self.summaries()

    def test_cross_chunk_metadata_mismatch(self):
        self.write_chunk(1, 4, ("dense",))
        path = self.write_chunk(1, 4, ("face",)).with_suffix(".replicates.csv")
        self.change(path, lambda rows: rows[0].update(truth_eigenvalues="1.0;0.5;0.24"))
        with self.assert_invalid("paired routes have different data metadata"):
            self.summaries()

    def test_bad_chunk_aggregate(self):
        path = self.write_chunk()
        self.change(path, lambda rows: rows[0].update(face_wall_seconds_mean="999"))
        with self.assert_invalid("differs from retained records"):
            self.summaries()

    def test_missing_sidecar_record(self):
        path = self.write_chunk().with_suffix(".replicates.csv")
        self.change(path, lambda rows: rows.pop())
        with self.assert_invalid("missing 1 declared"):
            self.summaries()

    def test_duplicate_sidecar_record(self):
        path = self.write_chunk().with_suffix(".replicates.csv")
        self.change(path, lambda rows: rows.append(rows[0].copy()))
        with self.assert_invalid("duplicate or unexpected replicate/route"):
            self.summaries()

    def test_record_outside_declared_scope(self):
        path = self.write_chunk(1, 2).with_suffix(".replicates.csv")
        self.change(path, lambda rows: rows[0].update(replicate="3"))
        with self.assert_invalid("duplicate or unexpected replicate/route"):
            self.summaries()

    def test_incomplete_chunk_metadata(self):
        path = self.write_chunk()
        self.change(path, lambda rows: rows[0].pop("routes"))
        with self.assert_invalid("incomplete chunk metadata"):
            self.summaries()

    def test_aggregates_for_unselected_route(self):
        path = self.write_chunk(1, 4, ("face",))
        self.change(path, lambda rows: rows[0].update(dense_successes="4"))
        with self.assert_invalid("aggregates for unselected route"):
            self.summaries()

    def test_invalid_ranges_and_routes(self):
        path = self.write_chunk()
        original = read_rows(path)
        for updates in ({"first_replicate": "0"}, {"first_replicate": "5"},
                        {"last_replicate": "5"}, {"routes": ""},
                        {"routes": "face;face"}, {"routes": "pace"}):
            with self.subTest(updates=updates):
                result = dict(original[0], **updates)
                write_rows(path, [result])
                with self.assertRaises(ValueError):
                    self.summaries()

    def test_new_diagnostics_require_type_column(self):
        path = self.write_chunk().with_suffix(".failures.csv")
        write_rows(path, [], FAILURE_FIELDS)
        with self.assert_invalid("missing required columns"):
            self.summaries()

    def test_new_diagnostics_require_nonempty_type(self):
        path = self.write_chunk(failures=((1, "face"),)).with_suffix(".failures.csv")
        self.change(path, lambda rows: rows[0].update(error_type=" "))
        with self.assert_invalid("needs its exception type"):
            self.summaries()

    def test_missing_failure_diagnostic(self):
        path = self.write_chunk(failures=((1, "face"),)).with_suffix(".failures.csv")
        write_rows(path, [], FAILURE_FIELDS + ("error_type",))
        with self.assert_invalid("missing failed-fit diagnostics"):
            self.summaries()

    def test_unexpected_failure_diagnostic(self):
        path = self.write_chunk(failures=((1, "face"),)).with_suffix(".failures.csv")
        self.change(path, lambda rows: rows[0].update(replicate="2"))
        with self.assert_invalid("duplicate or unexpected failure"):
            self.summaries()

    def test_incorrect_failure_seed(self):
        path = self.write_chunk(failures=((1, "face"),)).with_suffix(".failures.csv")
        self.change(path, lambda rows: rows[0].update(seed="99"))
        with self.assert_invalid("incorrect failure seed"):
            self.summaries()

    def test_ignore_sidecar_names_and_nonresults(self):
        self.write_chunk()
        (self.directory / "summary.csv").write_text("not a result row\n")
        self.assertEqual(len(self.summaries()), 2)


if __name__ == "__main__":
    unittest.main()
