"""Token + learned absolute position (+ optional speaker) embeddings."""

from __future__ import annotations

from typing import Any

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="dialogue_models")
class TokenAndPositionEmbedding(keras.layers.Layer):
    """Sum of token, learned position and (optionally) speaker embeddings, then dropout.

    Args:
        vocab_size: number of token IDs.
        max_len: maximum sequence length (number of learned positions).
        d_model: embedding width.
        num_speakers: number of speakers; 0 disables the speaker embedding. Speaker ID 0 is
            reserved for "no speaker" (padding / special tokens), so the table has
            ``num_speakers + 1`` rows.
        dropout: dropout rate applied to the summed embedding.
        init_std: standard deviation of the truncated-normal initialiser.
    """

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
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_speakers = num_speakers
        self.dropout_rate = dropout
        self.init_std = init_std
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

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.token_embedding.build(input_shape)
        if self.speaker_embedding is not None:
            self.speaker_embedding.build(input_shape)
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.max_len, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=self.init_std),
        )
        super().build(input_shape)

    def call(self, token_ids, speaker_ids=None, training: bool | None = None):
        """Embed ``token_ids`` (batch, length); ``speaker_ids`` has the same shape."""
        length = ops.shape(token_ids)[-1]
        x = self.token_embedding(token_ids)
        x = x + ops.cast(self.position_embedding[:length], x.dtype)[None, :, :]
        if self.speaker_embedding is not None:
            if speaker_ids is None:
                raise ValueError("speaker_ids is required when num_speakers > 0")
            x = x + self.speaker_embedding(speaker_ids)
        return self.dropout(x, training=training)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return (*input_shape, self.d_model)

    def get_config(self) -> dict[str, Any]:
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
