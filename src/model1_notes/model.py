"""Model 1 v3 student: hierarchical meeting encoder with per-sentence heads (PLAN.md 10.4).

Inputs: ``token_ids`` (batch, S, T) int32 (one ``[CLS] ... [SEP]`` sentence per slot, all
``[PAD]`` for empty slots) and ``speaker_ids`` (batch, S) int32 (1..8, 0 = empty slot).

1. **Sentence encoder** (layer ``encoder``, a :class:`TransformerEncoder` with the MLM
   checkpoint's architecture) encodes every sentence separately; tokens carry their sentence's
   speaker ID, as in MLM pretraining.
2. The ``[CLS]`` state of each sentence + sentence-position, speaker and speaker-change
   embeddings feeds the **meeting encoder** (pre-LN blocks over the sentences, padding masked).
3. Heads (float32, stable under ``mixed_float16``):
       ``importance``  (batch, S)          logit
       ``zone``        (batch, S, 5)       logits, :data:`~src.model1_notes.data.ZONES`
       ``reason``      (batch, S, R)       logits, Qwen reason classes
       ``depends_on``  (batch, S, S + 1)   pointer logits over earlier sentences; the last
                                           column is "none"
       ``spans``       (batch, S, T, 5)    BIO owner/deadline logits per token
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
from keras import ops

from src.layers.transformer import TransformerEncoder, TransformerEncoderBlock
from src.model1_scorer.model import load_encoder_weights
from src.tokenizer.tokenizer import PAD_ID

NEG = -1e9
MODEL_KEYS = (
    "vocab_size",
    "max_len",
    "d_model",
    "num_layers",
    "meeting_layers",
    "num_heads",
    "ffn_dim",
    "dropout",
    "init_std",
    "num_speakers",
    "num_zones",
    "num_reasons",
    "num_span_tags",
)


@keras.saving.register_keras_serializable(package="dialogue_models")
class DependsOnPointer(keras.layers.Layer):
    """Pointer logits from each sentence to every earlier real sentence, plus a "none" column.

    ``logits[b, i, j] = q_i . k_j / sqrt(d)`` for ``j < i`` and sentence ``j`` real, else a
    large negative value; ``logits[b, i, S]`` is a learned "depends on nothing" score.
    """

    def __init__(self, d_model: int, init_std: float = 0.02, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.init_std = init_std
        init = keras.initializers.TruncatedNormal(stddev=init_std)
        self.query = keras.layers.Dense(d_model, kernel_initializer=init, name="query")
        self.key = keras.layers.Dense(d_model, kernel_initializer=init, name="key")
        self.none = keras.layers.Dense(1, kernel_initializer=init, name="none")

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        for layer in (self.query, self.key, self.none):
            layer.build(input_shape)
        super().build(input_shape)

    def call(self, hidden, sentence_mask):
        h = ops.cast(hidden, "float32")
        scores = ops.matmul(self.query(h), ops.swapaxes(self.key(h), 1, 2))
        scores = scores / ops.sqrt(ops.cast(self.d_model, "float32"))
        s = ops.shape(h)[1]
        earlier = ops.arange(s)[None, :] < ops.arange(s)[:, None]  # (S, S): j < i
        allowed = ops.logical_and(earlier[None, :, :], sentence_mask[:, None, :])
        scores = ops.where(allowed, scores, NEG)
        return ops.concatenate([scores, self.none(h)], axis=-1)

    def compute_output_shape(self, hidden_shape, sentence_mask_shape=None) -> tuple:
        s = hidden_shape[1]
        return (hidden_shape[0], s, None if s is None else s + 1)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "d_model": self.d_model, "init_std": self.init_std}


@keras.saving.register_keras_serializable(package="dialogue_models")
class NotesModel(keras.Model):
    """Sentence encoder + meeting encoder + heads (see module docstring)."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        num_layers: int,
        meeting_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_sentences: int,
        dropout: float = 0.1,
        init_std: float = 0.02,
        num_speakers: int = 8,
        num_zones: int = 5,
        num_reasons: int = 9,
        num_span_tags: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.args = {
            "vocab_size": vocab_size,
            "max_len": max_len,
            "d_model": d_model,
            "num_layers": num_layers,
            "meeting_layers": meeting_layers,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "max_sentences": max_sentences,
            "dropout": dropout,
            "init_std": init_std,
            "num_speakers": num_speakers,
            "num_zones": num_zones,
            "num_reasons": num_reasons,
            "num_span_tags": num_span_tags,
        }
        init = keras.initializers.TruncatedNormal(stddev=init_std)
        self.encoder = TransformerEncoder(
            vocab_size,
            max_len,
            d_model,
            num_layers,
            num_heads,
            ffn_dim,
            dropout,
            init_std,
            num_speakers,
            name="encoder",
        )
        self.sentence_position = keras.layers.Embedding(
            max_sentences, d_model, embeddings_initializer=init, name="sentence_position"
        )
        self.meeting_speaker = keras.layers.Embedding(
            num_speakers + 1, d_model, embeddings_initializer=init, name="meeting_speaker"
        )
        self.speaker_change = keras.layers.Embedding(
            2, d_model, embeddings_initializer=init, name="speaker_change"
        )
        self.meeting_dropout = keras.layers.Dropout(dropout)
        self.meeting_blocks = [
            TransformerEncoderBlock(
                d_model, num_heads, ffn_dim, dropout, init_std, name=f"meeting_block_{i}"
            )
            for i in range(meeting_layers)
        ]
        self.meeting_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="meeting_norm")

        def head(units: int, name: str) -> keras.layers.Dense:
            return keras.layers.Dense(units, kernel_initializer=init, dtype="float32", name=name)

        self.importance_head = head(1, "importance_head")
        self.zone_head = head(num_zones, "zone_head")
        self.reason_head = head(num_reasons, "reason_head")
        self.span_head = head(num_span_tags, "span_head")
        self.pointer = DependsOnPointer(d_model, init_std, dtype="float32", name="depends_on_head")

    def call(self, inputs, training: bool | None = None):
        token_ids = ops.cast(inputs["token_ids"], "int32")
        speakers = ops.cast(inputs["speaker_ids"], "int32")
        b, s, t = ops.shape(token_ids)
        real_tokens = ops.not_equal(token_ids, PAD_ID)
        token_speakers = ops.where(real_tokens, speakers[:, :, None], 0)
        hidden = self.encoder(
            ops.reshape(token_ids, (-1, t)), ops.reshape(token_speakers, (-1, t)), training=training
        )
        d = self.args["d_model"]
        tokens = ops.reshape(hidden, (b, s, t, d))
        sentence_mask = real_tokens[:, :, 0]

        previous = ops.concatenate([speakers[:, :1], speakers[:, :-1]], axis=1)
        first = ops.arange(s)[None, :] == 0
        change = ops.cast(ops.logical_or(first, ops.not_equal(speakers, previous)), "int32")
        x = tokens[:, :, 0, :]
        x = x + ops.cast(self.sentence_position(ops.arange(s)), x.dtype)[None, :, :]
        x = x + ops.cast(self.meeting_speaker(speakers), x.dtype)
        x = x + ops.cast(self.speaker_change(change), x.dtype)
        x = self.meeting_dropout(x, training=training)
        for block in self.meeting_blocks:
            x = block(x, padding_mask=sentence_mask, training=training)
        x = self.meeting_norm(x)

        return {
            "importance": ops.squeeze(self.importance_head(x), axis=-1),
            "zone": self.zone_head(x),
            "reason": self.reason_head(x),
            "depends_on": self.pointer(x, sentence_mask),
            "spans": self.span_head(tokens),
        }

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self.args}


def build_notes_model(
    model_cfg: dict[str, Any], max_sentences: int, sentence_tokens: int
) -> NotesModel:
    """Build (and run once, so every weight exists) a student from the ``model`` config."""
    model = NotesModel(**{k: model_cfg[k] for k in MODEL_KEYS}, max_sentences=max_sentences)
    model(
        {
            "token_ids": ops.ones((1, 2, sentence_tokens), dtype="int32"),
            "speaker_ids": ops.ones((1, 2), dtype="int32"),
        }
    )
    return model


def load_sentence_encoder(model: NotesModel, checkpoint: str | Path) -> None:
    """Copy the MLM ``encoder.keras`` weights into the sentence encoder (same architecture)."""
    load_encoder_weights(model, checkpoint)
