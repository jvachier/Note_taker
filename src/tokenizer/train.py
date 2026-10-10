"""Stage 3: train the shared uncased WordPiece tokenizer and report its statistics.

Usage::

    uv run python -m src.tokenizer.train --config configs/tokenizer.yaml

Writes ``tokenizer.json``, ``vocab.txt`` and ``manifest.json`` to ``paths.output_dir`` and the
report (subwords per word, ``[UNK]`` rate on the test sets) to ``paths.report_path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tokenizers import BertWordPieceTokenizer, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Normalizer

from src.config import load_config
from src.seed import set_seed
from src.tokenizer.corpus import iter_corpus, iter_source
from src.tokenizer.tokenizer import (
    SPECIAL_TOKENS,
    TOKENIZER_FILE,
    UNK_ID,
    VOCAB_FILE,
    check_special_tokens,
)

log = logging.getLogger(__name__)


def compute_alphabet(texts: Iterable[str], normalizer: Normalizer, limit: int) -> list[str]:
    """The ``limit`` most frequent non-space characters after normalisation.

    Ties are broken by the character itself, so the result only depends on the corpus. (The
    trainer's own ``limit_alphabet`` breaks ties in hash-map order.)
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(normalizer.normalize_str(text))
    for space in [c for c in counts if c.isspace()]:
        del counts[space]
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted(c for c, _ in ranked[:limit])


def train_tokenizer(make_texts: Callable[[], Iterable[str]], cfg: dict[str, Any]) -> Tokenizer:
    """Train a BERT-style WordPiece tokenizer with the settings in ``cfg``.

    ``make_texts`` returns a fresh iterator over the corpus; it is called twice (alphabet
    pass, then training).
    """
    if cfg["special_tokens"] != SPECIAL_TOKENS:
        raise ValueError("special_tokens in the config must match src.tokenizer.SPECIAL_TOKENS")
    wordpiece = BertWordPieceTokenizer(
        clean_text=cfg["clean_text"],
        handle_chinese_chars=cfg["handle_chinese_chars"],
        strip_accents=cfg["strip_accents"],
        lowercase=cfg["lowercase"],
        wordpieces_prefix=cfg["wordpieces_prefix"],
    )
    alphabet = compute_alphabet(make_texts(), wordpiece.normalizer, cfg["limit_alphabet"])
    wordpiece.train_from_iterator(
        make_texts(),
        vocab_size=cfg["vocab_size"],
        min_frequency=cfg["min_frequency"],
        limit_alphabet=len(alphabet),
        initial_alphabet=alphabet,
        special_tokens=cfg["special_tokens"],
        wordpieces_prefix=cfg["wordpieces_prefix"],
        show_progress=False,
    )
    tokenizer = wordpiece._tokenizer  # the underlying `tokenizers.Tokenizer`
    canonicalize_ids(tokenizer, cfg["wordpieces_prefix"])
    check_special_tokens(tokenizer)
    return tokenizer


def canonicalize_ids(tokenizer: Tokenizer, prefix: str) -> None:
    """Renumber the vocab deterministically: special tokens first, then tokens sorted.

    The trainer assigns IDs to the alphabet in hash-map order, so two runs on the same corpus
    give the same token set with different IDs. WordPiece encoding only depends on the
    set, so a canonical order makes the saved files reproducible without changing behaviour.
    """
    vocab = tokenizer.get_vocab()
    rest = sorted(set(vocab) - set(SPECIAL_TOKENS))
    ordered = {tok: i for i, tok in enumerate([*SPECIAL_TOKENS, *rest])}
    tokenizer.model = WordPiece(
        ordered, unk_token=SPECIAL_TOKENS[UNK_ID], continuing_subword_prefix=prefix
    )


def tokenizer_report(
    tokenizer: Tokenizer, texts: Iterable[str], batch_size: int = 1000
) -> dict[str, float]:
    """Subword tokens per word and ``[UNK]`` rate over ``texts``.

    A "word" is a unit of the tokenizer's own pre-tokeniser (whitespace and punctuation
    split), so ``"don't"`` counts as three words and ``","`` as one.
    """
    n_texts = n_words = n_tokens = n_unk = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal n_words, n_tokens, n_unk
        for enc in tokenizer.encode_batch(batch, add_special_tokens=False):
            n_tokens += len(enc.ids)
            n_unk += sum(i == UNK_ID for i in enc.ids)
            n_words += len({w for w in enc.word_ids if w is not None})
        batch.clear()

    for text in texts:
        batch.append(text)
        n_texts += 1
        if len(batch) == batch_size:
            flush()
    flush()
    return {
        "texts": n_texts,
        "words": n_words,
        "tokens": n_tokens,
        "subwords_per_word": round(n_tokens / max(n_words, 1), 4),
        "unk_rate": round(n_unk / max(n_tokens, 1), 6),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    """Entry point: train, save and report from a config file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    paths = cfg["paths"]
    raw_dir, out_dir = Path(paths["raw_dir"]), Path(paths["output_dir"])
    max_items = cfg.get("max_items_per_source")
    start = time.time()

    log.info("Training on %s (max %s items per source)", cfg["corpus"], max_items)
    tokenizer = train_tokenizer(lambda: iter_corpus(raw_dir, cfg["corpus"], max_items), cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / TOKENIZER_FILE))
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    (out_dir / VOCAB_FILE).write_text("".join(f"{tok}\n" for tok, _ in vocab), encoding="utf-8")
    log.info("Vocab size %d, saved to %s", tokenizer.get_vocab_size(), out_dir)

    report: dict[str, Any] = {"vocab_size": tokenizer.get_vocab_size()}
    for source in cfg["report_sources"]:
        report[source] = tokenizer_report(tokenizer, iter_source(raw_dir, source, max_items))
        log.info("%s: %s", source, report[source])
    report_path = Path(paths["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    raw_manifest = raw_dir / "manifest.json"
    manifest = {
        "stage": "tokenizer",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.time() - start, 1),
        "config": cfg,
        "raw_manifest_sha256": _sha256(raw_manifest) if raw_manifest.exists() else None,
        "tokenizer_sha256": _sha256(out_dir / TOKENIZER_FILE),
        "vocab_sha256": _sha256(out_dir / VOCAB_FILE),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Wrote %s and %s", out_dir / "manifest.json", report_path)


if __name__ == "__main__":
    main()
