"""Helper for streaming KaBLE prompts to the ChatGPT API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from openai import OpenAI

from run_experiments import Example


_PROMPT_FIELDS: tuple[str, ...] = ("prompt", "query", "question", "input")


def _guess_prompt(payload: Mapping[str, object]) -> str:
    """Best-effort extraction of the prompt text from the payload."""

    for field in _PROMPT_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(
        "Could not find a prompt field in the example. Looked for one of: "
        + ", ".join(_PROMPT_FIELDS)
    )


@dataclass
class ChatGPTRunner:
    """Callable wrapper that submits each prompt to the ChatGPT API."""

    model: str
    api_key: str | None = None
    output_path: Path | None = None

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "An OpenAI API key is required. Provide one via the api_key argument "
                "or the OPENAI_API_KEY environment variable."
            )

        output = self.output_path or Path("outputs") / f"{self.model}-responses.jsonl"
        self._output_path = output.expanduser().resolve()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._client = OpenAI(api_key=key)

    @property
    def output_path(self) -> Path:
        return self._output_path

    def __call__(self, example: Example) -> None:
        prompt = _guess_prompt(example.payload)

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        message = completion.choices[0].message.content if completion.choices else ""

        record = {
            "model": self.model,
            "prompt": prompt,
            "response": message,
            "example": example.to_json(),
            "usage": getattr(completion, "usage", None),
        }

        with self._output_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
