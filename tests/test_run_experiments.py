from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import run_experiments


class RunExperimentsTestCase(unittest.TestCase):
    def test_default_subject_is_biomedicine(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "biomed.jsonl"
            summary = run_experiments.run_experiments(output_path=output_path)

            self.assertEqual(summary.subjects, ("BioMedicine",))
            self.assertEqual(summary.total_examples, summary.subject_counts.get("BioMedicine"))

            with output_path.open("r", encoding="utf-8") as handle:
                subjects = {json.loads(line)["subject"] for line in handle if line.strip()}
            self.assertEqual(subjects, {"BioMedicine"})

    def test_custom_subjects(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "math.jsonl"
            summary = run_experiments.run_experiments(
                subjects=("Math",),
                output_path=output_path,
            )

            self.assertEqual(summary.subjects, ("Math",))
            self.assertEqual(summary.total_examples, summary.subject_counts.get("Math"))

            with output_path.open("r", encoding="utf-8") as handle:
                subjects = {json.loads(line)["subject"] for line in handle if line.strip()}
            self.assertEqual(subjects, {"Math"})

    def test_discover_subjects_includes_biomedicine(self) -> None:
        subjects = run_experiments.discover_subjects()
        self.assertIn("BioMedicine", subjects)
        self.assertGreater(len(subjects), 1)


if __name__ == "__main__":
    unittest.main()
