"""Shared pytest fixtures. Pins the Keras backend before any test imports keras."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir() -> Path:
    """Path to the repo's ``configs/`` directory."""
    return REPO_ROOT / "configs"


WIKI_PARAGRAPHS = [
    " The quick brown fox jumps over the lazy dog . It was a sunny day in Paris , France . ",
    " Mr. Smith met Dr. Jones in the U.S. in 2011 ; they talked about NASA and the BBC . ",
    " Why did the well @-@ known team lose ? Nobody knows , but the fans were angry ! ",
    " The museum holds 1 @,@ 000 paintings : most of them are from the 19th century . ",
]
DIALOGUES = [
    (["Hi, Mr. Smith. How are you today?", "I'm fine, thanks. And you?", "Not bad."]),
    (["Did you see the game last night?", "Yes, it was great!", "I can't believe they won."]),
]


def write_tiny_raw(raw_dir: Path, copies: int = 30) -> None:
    """Small WikiText-style and DialogSum-style raw files for offline pipeline tests."""
    from src.data.dialogsum import Dialogue
    from src.data.io import write_jsonl, write_lines

    for split in ("train", "validation", "test"):
        lines = []
        for i in range(copies if split == "train" else 4):
            lines += ["", f" = Article {split} {i} = ", "", *WIKI_PARAGRAPHS]
        write_lines(raw_dir / "wikitext" / f"{split}.txt", lines)
        n = copies if split == "train" else 3
        records = [
            Dialogue(f"{split}_{i}", split, [1, 2, 1], utts, ["s"], ["t"]).to_dict()
            for i in range(n)
            for utts in DIALOGUES
        ]
        write_jsonl(raw_dir / "dialogsum" / f"{split}.jsonl", records)


@pytest.fixture(scope="session")
def tiny_assets(tmp_path_factory) -> dict[str, Path]:
    """Raw data + a trained tiny tokenizer, shared by the pipeline tests."""
    from src.config import deep_merge, load_config
    from src.tokenizer.corpus import iter_corpus
    from src.tokenizer.train import train_tokenizer

    root = tmp_path_factory.mktemp("assets")
    raw = root / "raw"
    write_tiny_raw(raw)
    cfg = deep_merge(
        load_config(REPO_ROOT / "configs" / "tokenizer.yaml"),
        {"vocab_size": 400, "min_frequency": 1},
    )
    tok = train_tokenizer(lambda: iter_corpus(raw, ["wikitext_train", "dialogsum_train"]), cfg)
    (root / "tokenizer").mkdir()
    tok.save(str(root / "tokenizer" / "tokenizer.json"))
    return {"root": root, "raw": raw, "tokenizer": root / "tokenizer"}
