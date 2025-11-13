"""Streamlit interface for preparing KaBLE experiments.

This app wraps :func:`run_experiments.run_experiments` to provide an interactive
workflow that defaults to the BioMedicine subject, mirroring the behaviour of
the CLI helper. It exposes toggles for switching to other KaBLE subjects,
changing the output location, and previewing the filtered prompts. The app is
compatible with Cloud Run deployments by relying solely on environment
configuration (``$PORT``) supplied at runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import streamlit as st

from chatgpt_runner import ChatGPTRunner
from run_experiments import DEFAULT_SUBJECTS, RunSummary, discover_subjects, run_experiments

_DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "kable-dataset"
_PREVIEW_LIMIT = 5
_TEST_MODE_LIMIT = 5


@st.cache_data(show_spinner=False)
def _cached_subjects(dataset_dir: str) -> tuple[str, ...]:
    """Cache the discovered subjects for faster UI responsiveness."""

    path = Path(dataset_dir).expanduser()
    return discover_subjects(dataset_dir=path)


def _subject_help_text(subjects: Iterable[str]) -> str:
    """Generate helper text summarising the available subjects."""

    joined = ", ".join(subjects)
    return f"Available subjects: {joined}" if joined else "No subjects found."


def _preview_examples(path: Path, limit: int = _PREVIEW_LIMIT) -> list[dict[str, object]]:
    """Read up to ``limit`` examples from a JSONL file for preview."""

    examples: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > limit:
                break
            text = line.strip()
            if not text:
                continue
            examples.append(json.loads(text))
    return examples


def _ensure_port_configuration() -> None:
    """Configure Streamlit to honour the Cloud Run ``$PORT`` setting."""

    port = os.environ.get("PORT")
    if port:
        # Streamlit reads configuration from environment variables, so we set
        # the canonical value here. Cloud Run injects the PORT env variable at
        # runtime, ensuring the app binds correctly without a custom config file.
        os.environ.setdefault("STREAMLIT_SERVER_PORT", port)
        os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")


def _render_summary(summary: RunSummary) -> None:
    """Render a textual and tabular summary of the run results."""

    st.success(
        f"Prepared {summary.total_examples} prompt(s) across "
        f"{len(summary.subjects)} subject(s)."
    )

    st.write("### Subject breakdown")
    st.table(
        {
            "Subject": list(summary.subject_counts.keys()),
            "Prompts": list(summary.subject_counts.values()),
        }
    )

    with summary.output_path.open("rb") as handle:
        data = handle.read()

    st.download_button(
        label="Download filtered prompts",
        data=data,
        file_name=summary.output_path.name,
        mime="application/json",
    )

    preview = _preview_examples(summary.output_path)
    if preview:
        st.write(f"### Preview (first {len(preview)} prompts)")
        st.json(preview)


def main() -> None:
    _ensure_port_configuration()

    st.set_page_config(page_title="KaBLE Experiment Runner", layout="wide")
    st.title("KaBLE Experiment Runner")
    st.caption(
        "Prepare KaBLE prompts for evaluation. The default selection targets "
        "the BioMedicine subject as requested."
    )

    dataset_default = str(_DEFAULT_DATASET_DIR)
    dataset_input = st.text_input(
        "Dataset directory",
        value=dataset_default,
        help="Path containing the KaBLE JSONL files.",
    )
    dataset_dir = Path(dataset_input).expanduser()

    if not dataset_dir.exists():
        st.error(f"Dataset directory not found: {dataset_dir}")
        st.stop()

    subjects = _cached_subjects(str(dataset_dir))
    st.caption(_subject_help_text(subjects))

    default_selection = [subject for subject in subjects if subject in DEFAULT_SUBJECTS]
    select_all = st.toggle("Include all subjects", value=not default_selection)
    selected_subjects = None if select_all else st.multiselect(
        "Subjects",
        options=subjects,
        default=default_selection or list(DEFAULT_SUBJECTS),
        help=(
            "Choose which subjects to prepare. If no selection is made, the "
            "BioMedicine default will be used."
        ),
    )

    output_input = st.text_input(
        "Output path (optional)",
        value="",
        help=(
            "Where to store the filtered prompts. Leave blank to use the "
            "automatic outputs/<subjects>.jsonl location."
        ),
    )
    output_path = Path(output_input).expanduser() if output_input.strip() else None

    test_mode = st.toggle(
        "Test mode (limit to 5 prompts)",
        value=False,
        help=(
            "Process only the first five matching prompts. Useful for verifying "
            "a deployment before running the full dataset."
        ),
    )

    st.write("### ChatGPT integration (optional)")
    chatgpt_enabled = st.toggle(
        "Send prompts to the ChatGPT API",
        value=False,
        help=(
            "When enabled, each prepared prompt is forwarded to the configured "
            "OpenAI model so you can capture responses alongside the JSONL export."
        ),
    )

    chatgpt_model = "gpt-4o-mini"
    chatgpt_api_key = ""
    chatgpt_output_path: Path | None = None
    if chatgpt_enabled:
        chatgpt_api_key = st.text_input(
            "OpenAI API key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Used only to contact the ChatGPT API from this session.",
        )
        chatgpt_model = st.text_input(
            "Model name",
            value="gpt-4o-mini",
            help="Provide any ChatGPT-compatible model identifier.",
        )
        chatgpt_output_input = st.text_input(
            "Responses output path (optional)",
            value="",
            help=(
                "Path to store ChatGPT responses as JSONL. Leave blank to default "
                "to outputs/<model>-responses.jsonl."
            ),
        )
        if chatgpt_output_input.strip():
            chatgpt_output_path = Path(chatgpt_output_input).expanduser()

    run_clicked = st.button("Prepare prompts")

    if run_clicked:
        runner = None
        if chatgpt_enabled:
            try:
                runner = ChatGPTRunner(
                    model=chatgpt_model.strip() or "gpt-4o-mini",
                    api_key=chatgpt_api_key.strip() or None,
                    output_path=chatgpt_output_path,
                )
            except ValueError as error:
                st.error(str(error))
                st.stop()

        with st.spinner("Running experiment preparation..."):
            summary = run_experiments(
                dataset_dir=dataset_dir,
                subjects=selected_subjects,
                output_path=output_path,
                max_examples=_TEST_MODE_LIMIT if test_mode else None,
                runner=runner,
            )
        _render_summary(summary)

        if runner is not None:
            responses_path = runner.output_path
            if responses_path.exists():
                st.success(f"ChatGPT responses saved to {responses_path}")
                preview = _preview_examples(responses_path)
                if preview:
                    st.write(f"### ChatGPT preview (first {len(preview)} responses)")
                    st.json(preview)
            else:
                st.warning("No ChatGPT responses were recorded for this run.")


if __name__ == "__main__":
    main()
