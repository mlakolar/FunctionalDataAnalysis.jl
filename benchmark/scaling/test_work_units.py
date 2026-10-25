"""Focused work-list and launcher checks; no Julia fits or Slurm submission."""
import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from configs import FIELDS, configurations
from work_units import (UNIT_FIELDS, check_work_units, overrides,
                        read_configurations, work_units)


HERE = Path(__file__).resolve().parent


class WorkUnitsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scaling units ")
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

    def test_exact_coverage_and_override_precedence(self):
        route_sizes = overrides(["dense=4"], self.configs)
        cell_sizes = overrides(["1:dense=9"], self.configs, per_cell=True)
        units = list(work_units(self.configs, 10, route_sizes, cell_sizes))
        dense = [unit for unit in units if unit["route"] == "dense"]
        self.assertEqual([(u["first_replicate"], u["last_replicate"]) for u in dense],
                         [(1, 9), (10, 18), (19, 23)])
        self.assertEqual(dense[-1]["output"], "cell-001-dense-0019-0023.csv")
        self.assertEqual(check_work_units(self.write_units(units), self.configs), 8)
        self.assertEqual(units, list(work_units(self.configs, 10, route_sizes, cell_sizes)))

    def test_rejects_invalid_or_unused_block_overrides(self):
        for value in ["dense=0", "other=2", "dense=x", "dense=2=3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                overrides([value], self.configs)
        for value in ["1:pace=2", "3:dense=2", "0:dense=2"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                overrides([value], self.configs, per_cell=True)
        with self.assertRaises(ValueError):
            overrides(["dense=2", "dense=3"], self.configs)
        with self.assertRaises(ValueError):
            list(work_units(self.configs, 0))

    def test_rejects_missing_duplicate_and_mismatched_work(self):
        units = list(work_units(self.configs, 10))
        cases = [units[:-1], units+[units[0]],
                 [dict(units[0], output="../outside.csv")]+units[1:],
                 [dict(units[0], route="pace")]+units[1:],
                 [dict(units[0], last_replicate=24)]+units[1:]]
        for altered in cases:
            with self.subTest(altered=altered[0]), self.assertRaises(ValueError):
                check_work_units(self.write_units(altered), self.configs)

    def executable(self, name, source):
        path = self.directory / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
        path.chmod(0o755)
        return path

    def environment(self):
        env = dict(os.environ)
        env.update(FDA_PROJECT=str(HERE.parents[1]), SCALING_CONFIG=str(self.config_path),
                   SCALING_OUTPUT=str(self.directory / "results"),
                   PATH=str(self.directory / "bin")+os.pathsep+env["PATH"])
        return env

    def test_worker_quotes_paths_and_requests_one_julia_thread(self):
        capture = self.directory / "julia.json"
        self.executable("julia", "#!/usr/bin/env python3\nimport json, os, sys\n"
                        "with open(os.environ['JULIA_CAPTURE'], 'w') as f:\n"
                        " json.dump({'args': sys.argv[1:], 'threads': [os.environ[k] for k in "
                        "('JULIA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')]}, f)\n")
        env = self.environment()
        env["JULIA_CAPTURE"] = str(capture)
        output = "cell-001-dense-0011-0020.csv"
        subprocess.run(["bash", str(HERE / "worker.sh"), "1", "dense", "11", "20", output],
                       env=env, check=True)
        called = json.loads(capture.read_text())
        self.assertEqual(called["args"][2:],
                         [str(self.config_path), "1", str(Path(env["SCALING_OUTPUT"]) / output),
                          "11", "20", "dense"])
        self.assertEqual(called["threads"], ["1"]*4)
        capture.unlink()
        failed = subprocess.run(["bash", str(HERE / "worker.sh"), "1", "dense", "11", "20", "../bad.csv"], env=env)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(capture.exists())

    def test_driver_preserves_inputs_and_resumes_the_same_list(self):
        capture = self.directory / "parallel.json"
        self.executable("module", "#!/bin/bash\nexit 0\n")
        self.executable("parallel", "#!/usr/bin/env python3\nimport json, os, sys\n"
                        "with open(os.environ['PARALLEL_CAPTURE'], 'w') as f:\n"
                        " json.dump(sys.argv[1:], f)\n")
        units_path = self.write_units(list(work_units(self.configs, 10)))
        env = self.environment()
        env.update(JULIA_MODULE="test-julia", SCALING_UNITS=str(units_path),
                   SLURM_NTASKS="2", SLURM_CPUS_PER_TASK="1", PARALLEL_CAPTURE=str(capture))
        command = ["bash", str(HERE / "discovery.sbatch")]
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        called = json.loads(capture.read_text())
        self.assertEqual(called[called.index("--jobs")+1], "2")
        self.assertNotIn("--resume-failed", called)
        self.assertTrue(any("srun --exclusive --exact -N1 -n1 -c1" in arg for arg in called))
        result_dir = Path(env["SCALING_OUTPUT"])
        self.assertEqual((result_dir / "work-units.tsv").read_bytes(), units_path.read_bytes())
        (result_dir / "parallel-joblog.tsv").touch()
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        self.assertIn("--resume-failed", json.loads(capture.read_text()))
        capture.unlink()
        with self.config_path.open("a") as stream:
            stream.write("\n")
        failed = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Resume input changed", failed.stderr)
        self.assertFalse(capture.exists())


if __name__ == "__main__":
    unittest.main()
