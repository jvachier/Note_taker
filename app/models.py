"""Load both released models (local folder or Hugging Face Hub) and run them on raw text.

A model source is either a local folder or a Hub repo ID, in the layout written by
:mod:`src.publish.package`: ``model.keras``, ``tokenizer.json`` and ``config.json``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import src.model1_notes.losses  # noqa: E402,F401  (registers the custom objects)
import src.model1_notes.model  # noqa: E402,F401
import src.model2_cleanup.model  # noqa: E402,F401
from src.model1_notes.notes import Note, annotate  # noqa: E402
from src.model2_cleanup.infer import clean_text  # noqa: E402

FILES = ["model.keras", "tokenizer.json", "config.json"]
REPO_ROOT = Path(__file__).resolve().parent.parent
# Used when the environment variable is not set: the local release folder written by
# `python -m src.publish.package` if it exists, else the Hub repo `$HF_NAMESPACE/<name>`.
DEFAULT_SOURCES = {
    "MODEL2_SOURCE": "punctuation-casing-en-7m",
    "MODEL1_SOURCE": "meeting-notes-en-21m",
}


def source_from_env(variable: str) -> str:
    """``$variable`` if set, else the local release folder if present, else the Hub repo
    ``$HF_NAMESPACE/<name>``.

    Raises:
        ValueError: none of the three is available.
    """
    if value := os.environ.get(variable):
        return value
    name = DEFAULT_SOURCES[variable]
    if (REPO_ROOT / "release" / name).is_dir():
        return str(REPO_ROOT / "release" / name)
    if namespace := os.environ.get("HF_NAMESPACE"):
        return f"{namespace}/{name}"
    raise ValueError(
        f"No model for {variable}: set it to a folder or a Hub repo ID, set HF_NAMESPACE "
        f"(the Hub user or organisation that hosts {name}), or build release/{name} with "
        "`python -m src.publish.package`."
    )


def resolve(source: str, revision: str | None = None) -> Path:
    """Local folder for ``source``: itself if it is a folder, else a Hub download (cached)."""
    path = Path(source)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source, revision=revision, allow_patterns=FILES))


@dataclass
class Bundle:
    """One released model: weights, tokenizer and its ``config.json``."""

    model: keras.Model
    tokenizer: Tokenizer
    config: dict[str, Any]
    source: str

    @classmethod
    def load(cls, source: str, kind: str, revision: str | None = None) -> Bundle:
        folder = resolve(source, revision)
        config = json.loads((folder / "config.json").read_text())
        if config["kind"] != kind:
            raise ValueError(f"{source} holds a {config['kind']} model, expected {kind}")
        model = keras.saving.load_model(folder / "model.keras", compile=False)
        tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        return cls(model, tokenizer, config, source)

    @property
    def params(self) -> int:
        return int(self.model.count_params())


class Models:
    """Model 2 (cleanup) and Model 1 v3 (meeting notes), ready for raw text."""

    def __init__(self, cleanup: Bundle, notes: Bundle) -> None:
        self.cleanup = cleanup
        self.notes = notes

    @classmethod
    def from_env(cls) -> Models:
        """Sources from ``MODEL2_SOURCE`` / ``MODEL1_SOURCE`` (see :func:`source_from_env`)."""
        revision = os.environ.get("MODEL_REVISION") or None
        return cls(
            Bundle.load(source_from_env("MODEL2_SOURCE"), "cleanup", revision),
            Bundle.load(source_from_env("MODEL1_SOURCE"), "notes", revision),
        )

    def clean(self, text: str) -> tuple[str, float]:
        """Punctuated and cased ``text`` and the time it took (ms)."""
        settings = self.cleanup.config["inference"]
        start = time.perf_counter()
        out = clean_text(
            self.cleanup.model,
            self.cleanup.tokenizer,
            text,
            settings["window_words"],
            settings["max_len"],
        )
        return out, (time.perf_counter() - start) * 1000

    def annotate(
        self, speakers: list[str], sentences: list[str], threshold: float | None = None
    ) -> tuple[list[Note], float]:
        """Every sentence with its importance, zone, kept flag and owner / deadline (ms).

        ``threshold`` overrides the released keep threshold (the app's slider)."""
        settings = dict(self.notes.config["inference"])
        if threshold is not None:
            settings["threshold"] = threshold
        start = time.perf_counter()
        notes = annotate(self.notes.model, self.notes.tokenizer, speakers, sentences, settings)
        return notes, (time.perf_counter() - start) * 1000
