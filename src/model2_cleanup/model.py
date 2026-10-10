"""Model 2 architecture: shared encoder + casing and punctuation token-classification heads."""

from __future__ import annotations

from typing import Any

import keras

from src.layers.transformer import TransformerEncoder
from src.model2_cleanup.labels import CASING_CLASSES, PUNCT_CLASSES


def build_cleanup_model(model_cfg: dict[str, Any]) -> keras.Model:
    """Functional model ``token_ids -> {"casing": logits, "punct": logits}``.

    Heads run in float32 so logits stay stable under ``mixed_float16``.
    """
    init = keras.initializers.TruncatedNormal(stddev=model_cfg["init_std"])
    token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    hidden = TransformerEncoder(
        vocab_size=model_cfg["vocab_size"],
        max_len=model_cfg["max_len"],
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        init_std=model_cfg["init_std"],
        name="encoder",
    )(token_ids)
    casing = keras.layers.Dense(
        len(CASING_CLASSES), kernel_initializer=init, dtype="float32", name="casing"
    )(hidden)
    punct = keras.layers.Dense(
        len(PUNCT_CLASSES), kernel_initializer=init, dtype="float32", name="punct"
    )(hidden)
    return keras.Model(token_ids, {"casing": casing, "punct": punct}, name="cleanup")
