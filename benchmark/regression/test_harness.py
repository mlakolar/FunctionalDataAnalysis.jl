"""Focused work-list and summary checks on the committed smoke outputs; no Julia fits."""
import csv
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from configs import FIELDS, ROUTES, configurations
from summarize import read_configurations as read_summary_configurations
from summarize import summarize
from work_units import UNIT_FIELDS, check_work_units, read_configurations, work_units


HERE = Path(__file__).resolve().parent
SMOKE = HERE / "smoke"


def rewrite(path, edit):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows = edit(rows)
    with path.open("w", newline="") as stream:
        csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(rows)


class WorkUnitsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="regression units ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config_path = self.directory / "configurations.csv"
        with self.config_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            rows = list(configurations(smoke=True))
            rows[0]["replicates"], rows[1]["replicates"] = 23, 7
            writer.writerows(rows)
        self.configs = read_configurations(self.config_path)

    def write_units(self, units):
        path = self.directory / "work units.tsv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=UNIT_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(units)
        return path

    def test_exact_coverage(self):
        units = list(work_units(self.configs, 10))
        first_cell = [(u["first_replicate"], u["last_replicate"]) for u in units
                      if u["cell"] == 1 and u["route"] == "pfr"]
        self.assertEqual(first_cell, [(1, 10), (11, 20), (21, 23)])
        self.assertEqual(units[0]["output"], "cell-001-fpc-0001-0010.csv")
        self.assertEqual(check_work_units(self.write_units(units), self.configs),
                         len(ROUTES) * (3 + 1))

    def test_rejects_incomplete_or_overlapping_lists(self):
        units = list(work_units(self.configs, 10))
        bad = dict(units[0], last_replicate=11, output="cell-001-fpc-0001-0011.csv")
        renamed = dict(units[0], output="other.csv")
        for variant in (units[1:], units + [units[0]], [bad] + units[1:], [renamed] + units[1:]):
            with self.subTest(n=len(variant)), self.assertRaises(ValueError):
                check_work_units(self.write_units(variant), self.configs)
        with self.assertRaises(ValueError):
            list(work_units(self.configs, 0))

    def test_committed_grid_is_recreated(self):
        with (HERE / "configurations.csv").open(newline="") as stream:
            committed = list(csv.DictReader(stream))
        expected = [{k: str(v) for k, v in row.items()} for row in configurations()]
        self.assertEqual(committed, expected)
        self.assertEqual(len(committed), 24)
        self.assertTrue(all(row["replicates"] == "200" for row in committed))

    def test_launcher_syntax(self):
        for script in ("worker.sh", "discovery.sbatch"):
            subprocess.run(["bash", "-n", str(HERE / script)], check=True)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="regression summary ")
        self.addCleanup(self.temporary.cleanup)
        self.results = Path(self.temporary.name) / "results"
        self.results.mkdir()
        for path in SMOKE.glob("cell-*"):
            shutil.copy(path, self.results / path.name)
        self.configs = read_summary_configurations(SMOKE / "configurations.csv")

    def test_smoke_summary(self):
        rows = summarize(self.configs, self.results)
        self.assertEqual(len(rows), 2 * len(ROUTES))
        self.assertEqual({row["route"] for row in rows}, set(ROUTES))
        self.assertTrue(all(row["successes"] + row["failures"] == 2 for row in rows))

    def check_rejected(self, name, edit):
        rewrite(self.results / name, edit)
        with self.assertRaises(ValueError):
            summarize(self.configs, self.results)

    def test_missing_replicate(self):
        self.check_rejected("cell-001.replicates.csv", lambda rows: rows[:-1])

    def test_duplicate_replicate(self):
        self.check_rejected("cell-001.replicates.csv", lambda rows: rows + [rows[-1]])

    def test_wrong_seed(self):
        def edit(rows):
            rows[1][1] = str(int(rows[1][1]) + 1)
            return rows
        self.check_rejected("cell-002.replicates.csv", edit)

    def test_paired_grid_mismatch(self):
        def edit(rows):
            column = rows[0].index("grid_start")
            rows[-1][column] = "0.01"
            return rows
        self.check_rejected("cell-002.replicates.csv", edit)

    def test_aggregate_mismatch(self):
        def edit(rows):
            column = rows[0].index("pfr_beta_relative_l2_error_mean")
            rows[1][column] = str(float(rows[1][column]) * 1.01)
            return rows
        self.check_rejected("cell-001.csv", edit)

    def test_failure_without_diagnostic(self):
        def edit(rows):
            header = rows[0]
            for column in range(header.index("beta_relative_l2_error"), len(header)):
                if header[column] not in ("grid_start", "grid_end", "signal_variance"):
                    rows[1][column] = ""
            rows[1][header.index("status")] = "failure"
            return rows
        self.check_rejected("cell-001.replicates.csv", edit)

    def test_missing_cell(self):
        for path in self.results.glob("cell-002*"):
            path.unlink()
        with self.assertRaises(ValueError):
            summarize(self.configs, self.results)


if __name__ == "__main__":
    unittest.main()
