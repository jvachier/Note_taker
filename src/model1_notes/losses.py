"""Model 1 v3 losses (PLAN.md 10.5). Each returns one value per sentence (or token); padding
and missing labels are removed with ``sample_weight`` (``reduction="mean_with_sample_weight"``).

``NotesImportanceLoss`` packs the gold label and both teachers in the last axis of ``y_true``
(see :data:`IMPORTANCE_CHANNELS`); each teacher term is a binary KL at temperature ``T``
(scaled by ``T**2``) and only counts where that teacher labelled the sentence.

``ZoneLoss`` is a cross-entropy whose ``context`` / ``closing`` logits are removed for AMI
sentences, which AMI never annotates with those zones (PLAN.md 10.4).
"""

from __future__ import annotations

from typing import Any

import keras
from keras import ops

#: ``y_true[..., k]`` of :class:`NotesImportanceLoss`.
IMPORTANCE_CHANNELS = (
    "gold",
    "gold_weight",
    "qwen_p",
    "qwen_weight",
    "roberta_logit",
    "roberta_weight",
)
#: Zones that AMI never uses (indices in ``ZONES``: context, closing).
AMI_MASKED_ZONES = (0, 4)
_EPS = 1e-4
_NEG = -1e9


def _binary_kl(teacher_logit, student_logit, temperature: float):
    """``KL(Bern(sigmoid(t / T)) || Bern(sigmoid(s / T))) * T**2``."""
    p = ops.sigmoid(teacher_logit / temperature)
    log_q = ops.log_sigmoid(student_logit / temperature)
    log_1q = ops.log_sigmoid(-student_logit / temperature)
    p = ops.clip(p, _EPS, 1.0 - _EPS)
    kl = p * (ops.log(p) - log_q) + (1.0 - p) * (ops.log(1.0 - p) - log_1q)
    return kl * temperature**2


@keras.saving.register_keras_serializable(package="dialogue_models")
class NotesImportanceLoss(keras.losses.Loss):
    """``w_gold * conf * BCE(gold) + w_qwen * KL(Qwen) + w_roberta * KL(RoBERTa)`` per sentence."""

    def __init__(
        self,
        gold: float = 1.0,
        qwen: float = 0.0,
        roberta: float = 0.0,
        temperature: float = 2.0,
        name: str = "notes_importance_loss",
        reduction: str = "mean_with_sample_weight",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.gold, self.qwen, self.roberta, self.temperature = gold, qwen, roberta, temperature

    def call(self, y_true, y_pred):
        y = ops.cast(y_true, "float32")
        logit = ops.cast(y_pred, "float32")
        gold, gold_w, qwen_p, qwen_w, rob_logit, rob_w = (y[..., k] for k in range(6))
        target = ops.clip(gold, 0.0, 1.0)
        loss = self.gold * gold_w * ops.binary_crossentropy(target, logit, from_logits=True)
        if self.qwen:
            p = ops.clip(qwen_p, _EPS, 1.0 - _EPS)
            qwen_logit = ops.log(p) - ops.log(1.0 - p)
            loss += self.qwen * qwen_w * _binary_kl(qwen_logit, logit, self.temperature)
        if self.roberta:
            loss += self.roberta * rob_w * _binary_kl(rob_logit, logit, self.temperature)
        return loss

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "gold": self.gold,
            "qwen": self.qwen,
            "roberta": self.roberta,
            "temperature": self.temperature,
        }


@keras.saving.register_keras_serializable(package="dialogue_models")
class ZoneLoss(keras.losses.Loss):
    """Sparse CE on zone logits; ``y_true[..., 0]`` = zone, ``y_true[..., 1]`` = 1 for AMI."""

    def __init__(
        self,
        name: str = "zone_loss",
        reduction: str = "mean_with_sample_weight",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)

    def call(self, y_true, y_pred):
        y = ops.cast(y_true, "int32")
        logits = ops.cast(y_pred, "float32")
        n = ops.shape(logits)[-1]
        masked = ops.zeros((n,), dtype="bool")
        for k in AMI_MASKED_ZONES:
            masked = ops.logical_or(masked, ops.arange(n) == k)
        is_ami = ops.equal(y[..., 1], 1)[..., None]
        logits = ops.where(ops.logical_and(is_ami, masked), _NEG, logits)
        labels = ops.clip(y[..., 0], 0, n - 1)
        return ops.sparse_categorical_crossentropy(labels, logits, from_logits=True)
