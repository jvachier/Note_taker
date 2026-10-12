"""Self-contained code to load and run punctuation-casing-en-7m (punctuation and casing).

Only needs ``keras>=3.6``, ``jax`` and ``tokenizers``. Importing this module registers the
model's custom layers (package ``dialogue_models``) so ``model.keras`` loads::

    import modeling
    cleaner = modeling.Cleaner.from_folder(".")      # the downloaded repo folder
    cleaner("hi sarah are you free on friday")       # -> "Hi Sarah, are you free on Friday?"

The model reads lowercase words without punctuation and predicts, for each word, its casing
(lower / Capitalised / UPPER) and the punctuation that follows it (none / , / . / ?).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import ops  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PAD_ID, CLS_ID, SEP_ID = 0, 2, 3
LOWER, CAPITALISED, UPPER = 0, 1, 2
NONE, COMMA, PERIOD, QUESTION = 0, 1, 2, 3
PUNCT_TEXT = {COMMA: ",", PERIOD: ".", QUESTION: "?"}

# ----------------------------------------------------------------------------- layers


@keras.saving.register_keras_serializable(package="dialogue_models")
class TokenAndPositionEmbedding(keras.layers.Layer):
    """Sum of token, learned position and (optionally) speaker embeddings, then dropout."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        num_speakers: int = 0,
        dropout: float = 0.1,
        init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size, self.max_len, self.d_model = vocab_size, max_len, d_model
        self.num_speakers, self.dropout_rate, self.init_std = num_speakers, dropout, init_std
        init = keras.initializers.TruncatedNormal(stddev=init_std)
        self.token_embedding = keras.layers.Embedding(
            vocab_size, d_model, embeddings_initializer=init, name="token_embedding"
        )
        self.speaker_embedding = (
            keras.layers.Embedding(
                num_speakers + 1, d_model, embeddings_initializer=init, name="speaker_embedding"
            )
            if num_speakers
            else None
        )
        self.dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        self.token_embedding.build(input_shape)
        if self.speaker_embedding is not None:
            self.speaker_embedding.build(input_shape)
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.max_len, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=self.init_std),
        )
        super().build(input_shape)

    def call(self, token_ids, speaker_ids=None, training=None):
        length = ops.shape(token_ids)[-1]
        x = self.token_embedding(token_ids)
        x = x + ops.cast(self.position_embedding[:length], x.dtype)[None, :, :]
        if self.speaker_embedding is not None:
            x = x + self.speaker_embedding(speaker_ids)
        return self.dropout(x, training=training)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.d_model)

    def get_config(self):
        config = super().get_config()
        config.update(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            d_model=self.d_model,
            num_speakers=self.num_speakers,
            dropout=self.dropout_rate,
            init_std=self.init_std,
        )
        return config


@keras.saving.register_keras_serializable(package="dialogue_models")
class TransformerEncoderBlock(keras.layers.Layer):
    """Pre-LayerNorm block: ``x + MHA(LN(x))`` then ``x + FFN(LN(x))``."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ffn_dim = d_model, num_heads, ffn_dim
        self.dropout_rate, self.init_std = dropout, init_std
        init = keras.initializers.TruncatedNormal(stddev=init_std)
        self.attn_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="attn_norm")
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            kernel_initializer=init,
            name="attention",
        )
        self.attn_dropout = keras.layers.Dropout(dropout)
        self.ffn_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")
        self.ffn_in = keras.layers.Dense(
            ffn_dim, activation="gelu", kernel_initializer=init, name="ffn_in"
        )
        self.ffn_out = keras.layers.Dense(d_model, kernel_initializer=init, name="ffn_out")
        self.ffn_dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        self.attn_norm.build(input_shape)
        self.attention.build(input_shape, input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn_in.build(input_shape)
        self.ffn_out.build((*input_shape[:-1], self.ffn_dim))
        super().build(input_shape)

    def call(self, x, padding_mask=None, training=None):
        mask = None
        if padding_mask is not None:
            mask = ops.logical_and(padding_mask[:, :, None], padding_mask[:, None, :])
        h = self.attn_norm(x)
        h = self.attention(h, h, attention_mask=mask, training=training)
        x = x + self.attn_dropout(h, training=training)
        h = self.ffn_out(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_dropout(h, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_rate,
            init_std=self.init_std,
        )
        return config


@keras.saving.register_keras_serializable(package="dialogue_models")
class TransformerEncoder(keras.layers.Layer):
    """Embeddings, ``num_layers`` pre-LN blocks and a final LayerNorm; token 0 is padding."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        init_std: float = 0.02,
        num_speakers: int = 0,
        pad_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size, self.max_len, self.d_model = vocab_size, max_len, d_model
        self.num_layers, self.num_heads, self.ffn_dim = num_layers, num_heads, ffn_dim
        self.dropout_rate, self.init_std = dropout, init_std
        self.num_speakers, self.pad_id = num_speakers, pad_id
        self.embedding = TokenAndPositionEmbedding(
            vocab_size, max_len, d_model, num_speakers, dropout, init_std, name="embedding"
        )
        self.blocks = [
            TransformerEncoderBlock(
                d_model, num_heads, ffn_dim, dropout, init_std, name=f"block_{i}"
            )
            for i in range(num_layers)
        ]
        self.final_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")

    def build(self, input_shape):
        self.embedding.build(input_shape)
        hidden = (*input_shape, self.d_model)
        for block in self.blocks:
            block.build(hidden)
        self.final_norm.build(hidden)
        super().build(input_shape)

    def call(self, token_ids, speaker_ids=None, training=None):
        padding_mask = ops.not_equal(token_ids, self.pad_id)
        x = self.embedding(token_ids, speaker_ids, training=training)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, training=training)
        return self.final_norm(x)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.d_model)

    def get_config(self):
        config = super().get_config()
        config.update(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_rate,
            init_std=self.init_std,
            num_speakers=self.num_speakers,
            pad_id=self.pad_id,
        )
        return config


# ----------------------------------------------------------------------------- text

_EDGE_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_INNER_RE = re.compile(r"(?<=[^\W_])['’\-](?=[^\W_])|(?<=\d)[.,](?=\d)")


def input_form(token: str) -> str:
    """Model input form of a whitespace token: lowercase, edge punctuation removed.

    Apostrophes / hyphens between letters or digits and ``.``/``,`` between digits are kept
    ("don't", "1,000"). Returns ``""`` for tokens without letters or digits.
    """
    core = _EDGE_RE.match(token).group(2)
    kept = {m.start() for m in _INNER_RE.finditer(core)}
    return "".join(c for i, c in enumerate(core) if c.isalnum() or i in kept).lower()


def input_words(text: str) -> list[str]:
    """Model input forms of the words of ``text`` (tokens without letters or digits dropped)."""
    return [f for f in (input_form(t) for t in text.split()) if f]


def render_word(form: str, casing: int, punct: int) -> str:
    """``form`` with its predicted casing and trailing punctuation applied."""
    if casing == UPPER:
        form = form.upper()
    elif casing == CAPITALISED:
        form = form[:1].upper() + form[1:]
    return form + PUNCT_TEXT.get(punct, "")


def window_starts(n_words: int, window_words: int) -> list[int]:
    """Start of each window: one window if it fits, else stride ``window_words // 2``."""
    if n_words <= window_words:
        return [0]
    stride = max(1, window_words // 2)
    return list(range(0, n_words - window_words, stride)) + [n_words - window_words]


# ----------------------------------------------------------------------------- inference


class Cleaner:
    """Restore punctuation and casing in lowercase, unpunctuated English text.

    Long texts are cut into overlapping windows of ``window_words`` words; every word takes its
    prediction from the window where it is farthest from both edges (training windows always
    end at a sentence end, so window edges are biased towards periods).
    """

    def __init__(self, model: keras.Model, tokenizer: Tokenizer, window_words: int, max_len: int):
        self.model, self.tokenizer = model, tokenizer
        self.window_words, self.max_len = window_words, max_len

    @classmethod
    def from_folder(cls, folder: str | Path) -> Cleaner:
        """Load ``model.keras``, ``tokenizer.json`` and ``config.json`` from a folder."""
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        model = keras.saving.load_model(folder / "model.keras", compile=False)
        tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        return cls(model, tokenizer, **config["inference"])

    def _encode(self, windows: Sequence[Sequence[str]]):
        token_ids = np.full((len(windows), self.max_len), PAD_ID, dtype=np.int32)
        positions = []
        encodings = self.tokenizer.encode_batch(
            [list(w) for w in windows], is_pretokenized=True, add_special_tokens=False
        )
        budget = self.max_len - 2
        for row, (window, enc) in enumerate(zip(windows, encodings, strict=True)):
            ids = enc.ids[:budget]
            token_ids[row, 0] = CLS_ID
            token_ids[row, 1 : 1 + len(ids)] = ids
            token_ids[row, 1 + len(ids)] = SEP_ID
            first: list[int | None] = [None] * len(window)
            for pos, w in enumerate(enc.word_ids[:budget]):
                if w is not None and first[w] is None:
                    first[w] = pos + 1
            positions.append(first)
        return token_ids, positions

    def __call__(self, text: str) -> str:
        """Punctuated and cased version of ``text`` (one output word per input word)."""
        words = input_words(text)
        if not words:
            return ""
        starts = window_starts(len(words), self.window_words)
        windows = [words[s : s + self.window_words] for s in starts]
        token_ids, positions = self._encode(windows)
        out = self.model.predict_on_batch(token_ids)
        casing = np.argmax(np.asarray(out["casing"]), axis=-1)
        punct = np.argmax(np.asarray(out["punct"]), axis=-1)
        result = []
        for j, form in enumerate(words):
            row = max(
                (r for r, s in enumerate(starts) if s <= j < s + len(windows[r])),
                key=lambda r: min(j - starts[r], starts[r] + len(windows[r]) - 1 - j),
            )
            pos = positions[row][j - starts[row]]
            if pos is None:  # cut off by max_len: keep the input form
                result.append(form)
            else:
                result.append(render_word(form, int(casing[row, pos]), int(punct[row, pos])))
        return " ".join(result)
