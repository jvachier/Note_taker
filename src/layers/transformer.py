"""Pre-LayerNorm transformer encoder block and the full encoder stack."""

from __future__ import annotations

from typing import Any

import keras
from keras import ops

from src.layers.embeddings import TokenAndPositionEmbedding


def _dense_init(std: float) -> keras.initializers.Initializer:
    return keras.initializers.TruncatedNormal(stddev=std)


def attention_mask_from_padding(padding_mask):
    """(batch, length) bool -> (batch, length, length) mask: queries attend to real keys."""
    return ops.logical_and(padding_mask[:, :, None], padding_mask[:, None, :])


@keras.saving.register_keras_serializable(package="dialogue_models")
class TransformerEncoderBlock(keras.layers.Layer):
    """Pre-LN block: ``x + MHA(LN(x))`` then ``x + FFN(LN(x))``, with dropout on both branches.

    The FFN is ``Dense(ffn_dim) -> GELU -> Dense(d_model)``.
    """

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
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.init_std = init_std
        self.attn_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="attn_norm")
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            kernel_initializer=_dense_init(init_std),
            name="attention",
        )
        self.attn_dropout = keras.layers.Dropout(dropout)
        self.ffn_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")
        self.ffn_in = keras.layers.Dense(
            ffn_dim, activation="gelu", kernel_initializer=_dense_init(init_std), name="ffn_in"
        )
        self.ffn_out = keras.layers.Dense(
            d_model, kernel_initializer=_dense_init(init_std), name="ffn_out"
        )
        self.ffn_dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.attn_norm.build(input_shape)
        self.attention.build(input_shape, input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn_in.build(input_shape)
        self.ffn_out.build((*input_shape[:-1], self.ffn_dim))
        super().build(input_shape)

    def call(self, x, padding_mask=None, training: bool | None = None):
        """``x``: (batch, length, d_model); ``padding_mask``: (batch, length) bool, True = real."""
        mask = None if padding_mask is None else attention_mask_from_padding(padding_mask)
        h = self.attn_norm(x)
        h = self.attention(h, h, attention_mask=mask, training=training)
        x = x + self.attn_dropout(h, training=training)
        h = self.ffn_out(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_dropout(h, training=training)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return input_shape

    def get_config(self) -> dict[str, Any]:
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
    """Embeddings, ``num_layers`` pre-LN blocks and a final LayerNorm.

    Token ID 0 (``[PAD]``) is treated as padding. Returns (batch, length, d_model).
    """

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
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.init_std = init_std
        self.num_speakers = num_speakers
        self.pad_id = pad_id
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

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.embedding.build(input_shape)
        hidden = (*input_shape, self.d_model)
        for block in self.blocks:
            block.build(hidden)
        self.final_norm.build(hidden)
        super().build(input_shape)

    def call(self, token_ids, speaker_ids=None, training: bool | None = None):
        padding_mask = ops.not_equal(token_ids, self.pad_id)
        x = self.embedding(token_ids, speaker_ids, training=training)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, training=training)
        return self.final_norm(x)

    def compute_output_shape(self, input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        return (*input_shape, self.d_model)

    def get_config(self) -> dict[str, Any]:
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
