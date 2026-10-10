"""Student importance loss (PLAN.md section 5.5).

Per dialogue: binary cross-entropy of the importance logits against the soft targets (mean
over labelled utterances) plus ``lambda`` x listwise KL between the teacher and student score
distributions over the utterances of the dialogue::

    KL( softmax(scale * target / T) || softmax(logit / T) )

Targets are in [0, 1] (teacher scores / 10, or oracle 0/1); ``scale`` puts them back on the
teacher's 0-10 scale so the target distribution is not flattened. Unlabelled utterances
(target ``-1``: padding, dropped utterances) are excluded from both terms.
"""

from __future__ import annotations

from typing import Any

import keras
from keras import ops

_NEG = -1e9


@keras.saving.register_keras_serializable(package="dialogue_models")
class ImportanceLoss(keras.losses.Loss):
    """BCE with soft targets + ``listwise_lambda`` x listwise KL; returns one value per dialogue."""

    def __init__(
        self,
        listwise_lambda: float = 1.0,
        temperature: float = 2.0,
        score_scale: float = 10.0,
        name: str = "importance_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.listwise_lambda = listwise_lambda
        self.temperature = temperature
        self.score_scale = score_scale

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        logits = ops.cast(y_pred, "float32")
        mask = ops.cast(y_true >= 0, "float32")
        n = ops.maximum(ops.sum(mask, axis=-1), 1.0)
        target = ops.where(mask > 0, y_true, 0.0)
        bce = ops.binary_crossentropy(target, logits, from_logits=True)
        bce = ops.sum(bce * mask, axis=-1) / n

        t = self.temperature
        p_log = ops.log_softmax(ops.where(mask > 0, self.score_scale * target / t, _NEG), axis=-1)
        q_log = ops.log_softmax(ops.where(mask > 0, logits / t, _NEG), axis=-1)
        kl = ops.sum(ops.exp(p_log) * (p_log - q_log) * mask, axis=-1)
        return bce + self.listwise_lambda * kl

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "listwise_lambda": self.listwise_lambda,
            "temperature": self.temperature,
            "score_scale": self.score_scale,
        }
