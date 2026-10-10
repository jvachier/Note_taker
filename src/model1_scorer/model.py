"""Model 1 student (PLAN.md section 5.2): encoder + per-utterance importance and tag heads.

Inputs ``token_ids`` / ``speaker_ids`` (batch, length) and ``cls_positions`` (batch, n_utt).
The hidden state at each utterance's ``[CLS]`` feeds
    ``importance``  Dense(1) logit per utterance (sigmoid -> score in [0, 1]), (batch, n_utt)
    ``tags``        Dense(4) logits per utterance (only when ``use_tags``), (batch, n_utt, 4)
Heads run in float32 so the outputs stay stable under ``mixed_float16``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras

from src.layers.heads import GatherPositions
from src.layers.transformer import TransformerEncoder
from src.model1_scorer.data import TAGS
from src.model1_scorer.mlm_model import ENCODER_KEYS


def build_student(model_cfg: dict[str, Any], use_tags: bool) -> keras.Model:
    """Functional student model (random initialisation)."""
    init = keras.initializers.TruncatedNormal(stddev=model_cfg["init_std"])
    token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    speaker_ids = keras.Input(shape=(None,), dtype="int32", name="speaker_ids")
    cls_positions = keras.Input(shape=(None,), dtype="int32", name="cls_positions")
    encoder = TransformerEncoder(**{k: model_cfg[k] for k in ENCODER_KEYS}, name="encoder")
    utterances = GatherPositions(name="gather_cls")(encoder(token_ids, speaker_ids), cls_positions)
    importance = keras.layers.Dense(
        1, kernel_initializer=init, dtype="float32", name="importance_logit"
    )(utterances)
    outputs = {
        "importance": keras.layers.Reshape((-1,), dtype="float32", name="importance")(importance)
    }
    if use_tags:
        outputs["tags"] = keras.layers.Dense(
            len(TAGS), kernel_initializer=init, dtype="float32", name="tags"
        )(utterances)
    inputs = {"token_ids": token_ids, "speaker_ids": speaker_ids, "cls_positions": cls_positions}
    return keras.Model(inputs, outputs, name="student")


def load_encoder_weights(student: keras.Model, checkpoint: str | Path) -> None:
    """Copy the encoder of an MLM ``encoder.keras`` into ``student``.

    The architecture must match (dropout may differ: it has no weights).
    """
    pretrained = keras.saving.load_model(checkpoint).get_layer("encoder")
    target = student.get_layer("encoder")
    keys = [k for k in ENCODER_KEYS if k != "dropout"]
    have, want = pretrained.get_config(), target.get_config()
    if diff := {k: (have[k], want[k]) for k in keys if have[k] != want[k]}:
        raise ValueError(f"Encoder architecture mismatch (checkpoint, student): {diff}")
    target.set_weights(pretrained.get_weights())
