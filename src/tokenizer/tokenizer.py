"""Loading the shared WordPiece tokenizer and encoding words with first-subword alignment.

Special-token IDs are a fixed file-format contract (checked on load and in tests against
``configs/tokenizer.yaml``): ``[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4, [S1]..[S8]=5..12``.

The tokenizer has no post-processor: callers add ``[CLS]`` / ``[SEP]`` explicitly, because the
two models use different layouts (Model 1 has one ``[CLS] ... [SEP]`` pair per utterance).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4
NUM_SPEAKERS = 8
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
    f"[S{i}]" for i in range(1, NUM_SPEAKERS + 1)
]
TOKENIZER_FILE = "tokenizer.json"
VOCAB_FILE = "vocab.txt"


def speaker_token_id(speaker: int) -> int:
    """Token ID of ``[S<speaker>]`` for a 1-based speaker number."""
    if not 1 <= speaker <= NUM_SPEAKERS:
        raise ValueError(f"Speaker {speaker} outside 1..{NUM_SPEAKERS}")
    return MASK_ID + speaker


def check_special_tokens(tokenizer: Tokenizer) -> None:
    """Raise ``ValueError`` if special tokens do not have their fixed IDs."""
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise ValueError(f"{token} has ID {actual}, expected {expected_id}")


def load_tokenizer(directory: str | Path) -> Tokenizer:
    """Load ``tokenizer.json`` from ``directory`` and check the special-token IDs."""
    tokenizer = Tokenizer.from_file(str(Path(directory) / TOKENIZER_FILE))
    check_special_tokens(tokenizer)
    return tokenizer


@dataclass
class WordEncoding:
    """Subword encoding of a list of words.

    Attributes:
        ids: subword token IDs (including ``[CLS]`` / ``[SEP]`` if requested).
        word_ids: for each token, the index of its source word, or ``None`` for specials.
        first_subword: for each token, ``True`` if it is the first subword of its word.
        word_to_token: for each word, the position of its first subword, or ``None`` if the
            word produced no token (e.g. it contained only control characters).
    """

    ids: list[int]
    word_ids: list[int | None]
    first_subword: list[bool]
    word_to_token: list[int | None]


def encode_words(
    tokenizer: Tokenizer, words: Sequence[str], add_cls_sep: bool = False
) -> WordEncoding:
    """Encode pre-split words, keeping the word-to-first-subword mapping.

    Each word may still be split further (``"don't"`` -> ``don ' t``); all its pieces share
    the same word index and only the first one has ``first_subword=True``.
    """
    enc = tokenizer.encode(list(words), is_pretokenized=True, add_special_tokens=False)
    ids = list(enc.ids)
    word_ids: list[int | None] = list(enc.word_ids)
    if add_cls_sep:
        ids = [CLS_ID, *ids, SEP_ID]
        word_ids = [None, *word_ids, None]
    first_subword = [
        w is not None and (i == 0 or word_ids[i - 1] != w) for i, w in enumerate(word_ids)
    ]
    word_to_token: list[int | None] = [None] * len(words)
    for i, (w, first) in enumerate(zip(word_ids, first_subword, strict=True)):
        if first:
            word_to_token[w] = i
    return WordEncoding(ids, word_ids, first_subword, word_to_token)
