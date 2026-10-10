"""MLM pretraining model for the Model 1 encoder (PLAN.md sections 3 and 5.1).

``MLMPretrainer`` = the shared :class:`~src.layers.transformer.TransformerEncoder` (with speaker
embeddings) + :class:`MLMHead`. The head scores only the masked positions (gathered by index,
so the 16k-way output layer runs on ~15% of the tokens) and its output projection is tied to
the token embedding matrix. After pretraining, :func:`encoder_model` wraps the encoder alone
as a functional model; that is what the students load (``encoder.keras``).
"""

from __future__ import annotations

from typing import Any

import keras
from keras import ops

from src.layers.transformer import TransformerEncoder

ENCODER_KEYS = (
    "vocab_size",
    "max_len",
    "d_model",
    "num_layers",
    "num_heads",
    "ffn_dim",
    "dropout",
    "init_std",
    "num_speakers",
)


@keras.saving.register_keras_serializable(package="dialogue_models")
class MLMHead(keras.layers.Layer):
    """Gather masked positions, ``Dense -> GELU -> LayerNorm``, then tied output + bias.

    Call with ``(hidden, positions, embeddings)``: ``hidden`` (batch, length, d_model),
    ``positions`` (batch, n_pred) int, ``embeddings`` (vocab, d_model) the token embedding
    matrix. Returns logits (batch, n_pred, vocab). Use ``dtype="float32"`` under mixed precision.
    """

    def __init__(self, vocab_size: int, d_model: int, init_std: float = 0.02, **kwargs: Any):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.init_std = init_std
        self.transform = keras.layers.Dense(
            d_model,
            activation="gelu",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_std),
            name="transform",
        )
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

    def build(self, hidden_shape, positions_shape=None, embeddings_shape=None) -> None:
        self.transform.build(hidden_shape)
        self.norm.build(hidden_shape)
        self.bias = self.add_weight(name="bias", shape=(self.vocab_size,), initializer="zeros")
        super().build(hidden_shape)

    def call(self, hidden, positions, embeddings):
        gathered = ops.take_along_axis(hidden, ops.cast(positions, "int32")[:, :, None], axis=1)
        h = self.norm(self.transform(gathered))
        return ops.matmul(h, ops.transpose(ops.cast(embeddings, h.dtype))) + self.bias

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(vocab_size=self.vocab_size, d_model=self.d_model, init_std=self.init_std)
        return config


@keras.saving.register_keras_serializable(package="dialogue_models")
class MLMPretrainer(keras.Model):
    """Encoder + tied MLM head.

    Inputs: dict with ``token_ids``, ``speaker_ids`` (batch, length) and ``masked_positions``
    (batch, n_pred). Output: logits (batch, n_pred, vocab).
    """

    def __init__(self, encoder_config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder_config = dict(encoder_config)
        self.encoder = TransformerEncoder(**self.encoder_config, name="encoder")
        self.head = MLMHead(
            self.encoder_config["vocab_size"],
            self.encoder_config["d_model"],
            self.encoder_config.get("init_std", 0.02),
            dtype="float32",
            name="mlm_head",
        )

    def call(self, inputs, training=None):
        hidden = self.encoder(inputs["token_ids"], inputs["speaker_ids"], training=training)
        embeddings = self.encoder.embedding.token_embedding.embeddings
        return self.head(hidden, inputs["masked_positions"], embeddings)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "encoder_config": self.encoder_config}


def build_mlm_model(model_cfg: dict[str, Any], batch: int = 2, n_pred: int = 2) -> MLMPretrainer:
    """Build (and run once, so all weights exist) an MLM model from the ``model`` config."""
    if not model_cfg.get("tie_embeddings", True):
        raise ValueError("Only tied MLM output embeddings are implemented (PLAN.md section 3).")
    model = MLMPretrainer({k: model_cfg[k] for k in ENCODER_KEYS}, name="mlm")
    length = model_cfg["max_len"]
    model(
        {
            "token_ids": ops.ones((batch, length), dtype="int32"),
            "speaker_ids": ops.zeros((batch, length), dtype="int32"),
            "masked_positions": ops.zeros((batch, n_pred), dtype="int32"),
        }
    )
    return model


def encoder_model(model: MLMPretrainer) -> keras.Model:
    """Functional ``{token_ids, speaker_ids} -> hidden`` model sharing the pretrained encoder."""
    token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    speaker_ids = keras.Input(shape=(None,), dtype="int32", name="speaker_ids")
    hidden = model.encoder(token_ids, speaker_ids)
    return keras.Model(
        {"token_ids": token_ids, "speaker_ids": speaker_ids}, hidden, name="mlm_encoder"
    )
