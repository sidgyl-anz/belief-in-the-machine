"""Utility script for running KaBLE experiments.

This module filters KaBLE tasks and prepares them for evaluation. By default,
only the **BioMedicine** subject is selected so that experiments focus on the
requested category.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

DATASET_DIR = Path(__file__).resolve().parent / "kable-dataset"
DEFAULT_SUBJECTS: tuple[str, ...] = ("BioMedicine",)


@dataclass(frozen=True)
class Example:
    """Represents a single KaBLE prompt from the JSONL files."""

    payload: Mapping[str, object]
    source_file: Path

    @property
    def subject(self) -> str:
        return str(self.payload["subject"])

    def to_json(self) -> dict[str, object]:
        data = dict(self.payload)
        data["source_file"] = str(self.source_file)
        return data


@dataclass(frozen=True)
class RunSummary:
    """Summary information returned after scheduling experiments."""

    total_examples: int
    subjects: tuple[str, ...]
    subject_counts: dict[str, int]
    output_path: Path

    def __str__(self) -> str:  # pragma: no cover - human readable helper
        subject_breakdown = ", ".join(
            f"{subject}: {count}" for subject, count in sorted(self.subject_counts.items())
        )
        return (
            f"Prepared {self.total_examples} examples "
            f"across {len(self.subjects)} subject(s) [{subject_breakdown}] "
            f"-> {self.output_path}"
        )


def iter_examples(
    *,
    dataset_dir: Path | None = None,
    subject_filter: Sequence[str] | None = None,
) -> Iterator[Example]:
    """Yield examples from the dataset, optionally filtering by subject."""

    dataset_root = dataset_dir or DATASET_DIR
    allowed_subjects = {subject.lower() for subject in subject_filter} if subject_filter else None

    for jsonl_path in sorted(dataset_root.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                subject = str(payload["subject"])
                if allowed_subjects is not None and subject.lower() not in allowed_subjects:
                    continue
                yield Example(payload=payload, source_file=jsonl_path)


def discover_subjects(dataset_dir: Path | None = None) -> tuple[str, ...]:
    """Return all available subjects in the dataset."""

    subjects: set[str] = set()
    for example in iter_examples(dataset_dir=dataset_dir):
        subjects.add(example.subject)
    return tuple(sorted(subjects))


def _build_output_path(
    *,
    subjects: Sequence[str] | None,
    output_path: Path | None,
) -> Path:
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        stem = "all-subjects"
    elif len(subjects) == 1:
        stem = subjects[0].lower()
    else:
        stem = "-".join(sorted(subject.lower() for subject in subjects))

    return output_dir / f"{stem}.jsonl"


def run_experiments(
    *,
    dataset_dir: Path | None = None,
    subjects: Sequence[str] | None = DEFAULT_SUBJECTS,
    output_path: Path | None = None,
    runner: Callable[[Example], None] | None = None,
) -> RunSummary:
    """Prepare experiments for the provided subjects.

    Parameters
    ----------
    dataset_dir:
        Directory containing the KaBLE JSONL files.
    subjects:
        Sequence of subject names to include. ``None`` selects all subjects.
    output_path:
        Optional path where the filtered prompts should be written as JSONL.
        Defaults to ``outputs/<subjects>.jsonl``.
    runner:
        Optional callable that will be invoked for each :class:`Example` to
        execute the actual model interaction. When ``None`` (the default), the
        function simply prepares the prompts without performing any inference.
    """

    dataset_root = dataset_dir or DATASET_DIR
    selected_subjects = None if subjects is None else tuple(subjects)
    output_file = _build_output_path(subjects=selected_subjects, output_path=output_path)

    examples: list[Example] = list(
        iter_examples(dataset_dir=dataset_root, subject_filter=selected_subjects)
    )
    counts = Counter(example.subject for example in examples)

    with output_file.open("w", encoding="utf-8") as handle:
        for example in examples:
            json.dump(example.to_json(), handle, ensure_ascii=False)
            handle.write("\n")
            if runner is not None:
                runner(example)

    summary = RunSummary(
        total_examples=len(examples),
        subjects=tuple(sorted(counts)),
        subject_counts=dict(counts),
        output_path=output_file,
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing KaBLE JSONL files (default: %(default)s).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help=(
            "Optional list of subject names to run. When omitted, only BioMedicine "
            "is selected. Use --all-subjects to process every subject."
        ),
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Ignore --subjects and include every available subject.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional JSONL path for the filtered prompts. Defaults to an automatic "
            "location inside the outputs/ directory."
        ),
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all subjects in the dataset and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_subjects:
        for subject in discover_subjects(dataset_dir=args.dataset_dir):
            print(subject)
        return 0

    if args.all_subjects and args.subjects:
        raise SystemExit("Cannot supply --subjects together with --all-subjects.")

    if args.all_subjects:
        selected_subjects = None
    elif args.subjects:
        selected_subjects = tuple(args.subjects)
    else:
        selected_subjects = DEFAULT_SUBJECTS

    summary = run_experiments(
        dataset_dir=args.dataset_dir,
        subjects=selected_subjects,
        output_path=args.output,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
