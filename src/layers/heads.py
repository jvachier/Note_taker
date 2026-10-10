"""Head helpers: gather the hidden states at given positions (e.g. each utterance's ``[CLS]``)."""

from __future__ import annotations

from typing import Any

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="dialogue_models")
class GatherPositions(keras.layers.Layer):
    """``(hidden (batch, length, d), positions (batch, n)) -> (batch, n, d)``."""

    def call(self, hidden, positions):
        return ops.take_along_axis(hidden, ops.cast(positions, "int32")[:, :, None], axis=1)

    def compute_output_shape(self, hidden_shape, positions_shape) -> tuple[int | None, ...]:
        return (*positions_shape, hidden_shape[-1])

    def get_config(self) -> dict[str, Any]:
        return super().get_config()
