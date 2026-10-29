"""RoBERTa-Large teacher and DistilBERT baseline for Model 1 v3 (PLAN.md 10.3, Notebook 1).

Both are pretrained encoders fine-tuned once on the corpus gold importance labels (KerasHub on
the JAX backend). They run **only** in the Model 1 teacher notebook; the student training code
never imports this module, and only their cached per-sentence logits reach Notebook 2.

**Input.** Consecutive sentences of a meeting are packed into windows of at most ``max_tokens``
tokens and ``max_sentences`` sentences: ``<bos> s1 <sep> s2 <sep> ...``; every sentence appears
in exactly one window (cut to ``max_sentence_tokens``). Each token carries the index of its
sentence in the window (``-1`` for specials and padding).

**Model.** ``backbone({"token_ids", "padding_mask"}) -> (B, L, d)``, then the mean of each
sentence's token states (:class:`SentenceMeanPool`) and ``Dense(1)``: one logit per sentence.
Loss: binary cross-entropy on the gold label, padding sentences weighted 0.

Tokenizers come from the Hugging Face ``tokenizers`` library (same vocabularies as the KerasHub
presets, checked on load); KerasHub's own tokenizers would need TensorFlow. Tests use a tiny
stand-in backbone and a toy tokenizer; the pretrained weights are only downloaded on Kaggle.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
from keras import ops

from src.training import build_optimizer

Encode = Callable[[str], list[int]]


@dataclass
class Specials:
    """Special token IDs of a teacher tokenizer."""

    bos: int
    sep: int
    pad: int


@dataclass
class Packed:
    """Packed windows: model inputs, labels and the sentence key of every label slot."""

    token_ids: np.ndarray  # (W, L) int32
    padding_mask: np.ndarray  # (W, L) bool
    sentence_index: np.ndarray  # (W, L) int32, -1 = no sentence
    labels: np.ndarray  # (W, K) float32, -1 = empty slot
    keys: list[list[tuple[str, int]]]  # per window: (note_id, sentence_id) per slot


def pack_meetings(
    meetings: dict[str, list[dict[str, Any]]],
    encode: Encode,
    specials: Specials,
    max_tokens: int,
    max_sentence_tokens: int,
    max_sentences: int,
) -> Packed:
    """Pack every meeting's sentences into windows (see module docstring).

    ``meetings`` maps a note ID to its rows (``sentence_id``, ``text``, ``importance``) in
    meeting order, as :func:`~src.model1_notes.data.read_corpus_split` returns them.
    """
    windows: list[tuple[list[int], list[int], list[float], list[tuple[str, int]]]] = []
    for note_id, rows in meetings.items():
        tokens, index, labels, keys = [specials.bos], [-1], [], []
        for row in rows:
            piece = encode(row["text"])[:max_sentence_tokens] or [specials.sep]
            full = len(tokens) + len(piece) + 1 > max_tokens or len(labels) == max_sentences
            if full and labels:
                windows.append((tokens, index, labels, keys))
                tokens, index, labels, keys = [specials.bos], [-1], [], []
            k = len(labels)
            tokens += [*piece, specials.sep]
            index += [k] * len(piece) + [-1]
            labels.append(float(row["importance"]))
            keys.append((note_id, int(row["sentence_id"])))
        if labels:
            windows.append((tokens, index, labels, keys))
    n = len(windows)
    token_ids = np.full((n, max_tokens), specials.pad, dtype=np.int32)
    sentence_index = np.full((n, max_tokens), -1, dtype=np.int32)
    label_arr = np.full((n, max_sentences), -1.0, dtype=np.float32)
    for w, (tokens, index, labels, _) in enumerate(windows):
        token_ids[w, : len(tokens)] = tokens
        sentence_index[w, : len(index)] = index
        label_arr[w, : len(labels)] = labels
    mask = np.zeros((n, max_tokens), dtype=bool)
    for w, (tokens, *_rest) in enumerate(windows):
        mask[w, : len(tokens)] = True
    return Packed(token_ids, mask, sentence_index, label_arr, [w[3] for w in windows])


@keras.saving.register_keras_serializable(package="dialogue_models")
class SentenceMeanPool(keras.layers.Layer):
    """``(hidden (B, L, d), sentence_index (B, L)) -> (B, K, d)``: mean token state per sentence."""

    def __init__(self, max_sentences: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_sentences = max_sentences

    def call(self, hidden, sentence_index):
        onehot = ops.one_hot(ops.cast(sentence_index, "int32"), self.max_sentences)  # -1 -> 0s
        onehot = ops.cast(onehot, hidden.dtype)
        summed = ops.einsum("blk,bld->bkd", onehot, hidden)
        counts = ops.maximum(ops.sum(onehot, axis=1), 1.0)[:, :, None]
        return summed / counts

    def compute_output_shape(self, hidden_shape, sentence_index_shape=None):
        return (hidden_shape[0], self.max_sentences, hidden_shape[-1])

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "max_sentences": self.max_sentences}


@keras.saving.register_keras_serializable(package="dialogue_models")
class SentenceBCE(keras.losses.Loss):
    """Binary cross-entropy on logits per sentence slot (empty slots weighted 0)."""

    def __init__(
        self, name: str = "sentence_bce", reduction: str = "mean_with_sample_weight", **kwargs: Any
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)

    def call(self, y_true, y_pred):
        return ops.binary_crossentropy(
            ops.cast(y_true, "float32"), ops.cast(y_pred, "float32"), from_logits=True
        )


def build_sentence_classifier(
    backbone: keras.Model, max_tokens: int, max_sentences: int
) -> keras.Model:
    """Backbone + per-sentence mean pooling + ``Dense(1)`` logit per sentence."""
    token_ids = keras.Input((max_tokens,), dtype="int32", name="token_ids")
    padding_mask = keras.Input((max_tokens,), dtype="bool", name="padding_mask")
    sentence_index = keras.Input((max_tokens,), dtype="int32", name="sentence_index")
    hidden = backbone({"token_ids": token_ids, "padding_mask": padding_mask})
    pooled = SentenceMeanPool(max_sentences, name="sentence_pool")(hidden, sentence_index)
    logits = keras.layers.Dense(1, dtype="float32", name="sentence_logit")(pooled)
    logits = keras.layers.Reshape((max_sentences,), dtype="float32", name="logits")(logits)
    inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
        "sentence_index": sentence_index,
    }
    return keras.Model(inputs, logits, name="sentence_classifier")


def inputs_of(packed: Packed) -> dict[str, np.ndarray]:
    return {
        "token_ids": packed.token_ids,
        "padding_mask": packed.padding_mask,
        "sentence_index": packed.sentence_index,
    }


def fine_tune(
    model: keras.Model,
    train: Packed,
    validation: Packed,
    cfg: dict[str, Any],
    seed: int,
) -> keras.callbacks.History:
    """Fine-tune on gold importance (BCE, empty slots weighted 0), keep the best validation."""
    keras.utils.set_random_seed(seed)
    steps = max(1, len(train.labels) // cfg["batch_size"]) * cfg["epochs"]
    optimizer = build_optimizer(
        {
            "name": "adamw",
            "schedule": "cosine",
            "peak_lr": cfg["peak_lr"],
            "warmup_fraction": cfg["warmup_fraction"],
            "weight_decay": cfg["weight_decay"],
        },
        steps,
    )
    model.compile(optimizer=optimizer, loss=SentenceBCE())

    def weights(p: Packed) -> np.ndarray:
        return (p.labels >= 0).astype(np.float32)

    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
    return model.fit(
        inputs_of(train),
        np.clip(train.labels, 0, 1),
        sample_weight=weights(train),
        validation_data=(
            inputs_of(validation),
            np.clip(validation.labels, 0, 1),
            weights(validation),
        ),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        callbacks=[stop],
        verbose=2,
    )


def predict_logits(model: keras.Model, packed: Packed, batch_size: int) -> list[dict[str, Any]]:
    """One row per sentence: ``note_id``, ``sentence_id``, ``logit``."""
    logits = np.asarray(model.predict(inputs_of(packed), batch_size=batch_size, verbose=0))
    rows = []
    for w, keys in enumerate(packed.keys):
        for k, (note_id, sentence_id) in enumerate(keys):
            rows.append(
                {"note_id": note_id, "sentence_id": sentence_id, "logit": float(logits[w, k])}
            )
    return rows


def write_logits(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Teacher table read by the student (``train.align_teacher``) and the evaluation."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [("note_id", pa.string()), ("sentence_id", pa.int32()), ("logit", pa.float32())]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path)


def hf_tokenizer(name: str, specials: dict[str, str]) -> tuple[Encode, Specials, int]:
    """``encode``, special IDs and vocabulary size of a Hugging Face tokenizer (teacher only)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(name)

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False).ids

    ids = {k: tok.token_to_id(v) for k, v in specials.items()}
    if None in ids.values():
        raise ValueError(f"{name}: missing special tokens {specials}")
    return encode, Specials(**ids), tok.get_vocab_size()


def find_local_preset(root: str | Path, preset: str) -> Path | None:
    """An attached KerasHub preset under ``root``: a directory holding ``config.json`` whose
    path contains ``preset`` (Kaggle mounts ``keras/<model>/keras/<preset>/<version>/``)."""
    root = Path(root)
    if not root.exists():
        return None
    found = sorted(
        (Path(d) for d, _, files in os.walk(root, followlinks=True) if "config.json" in files),
        key=lambda p: (preset not in p.parts, len(p.parts)),
    )
    return next((p for p in found if preset in p.parts), None)


def keras_hub_backbone(
    preset: str, vocab_size: int, search_root: str | Path | None = None
) -> keras.Model:  # pragma: no cover - Kaggle only
    """Pretrained KerasHub backbone; checks the vocabulary size.

    A non-interactive Kaggle session cannot attach models while it runs, so the preset must be
    attached as an input (Add Input -> Models); it is loaded from that local copy.
    """
    import keras_hub

    local = None if search_root is None else find_local_preset(search_root, preset)
    if search_root is not None and local is None:
        raise SystemExit(
            f"Attach the KerasHub model '{preset}' (Add Input -> Models -> Keras) and rerun."
        )
    backbone = keras_hub.models.Backbone.from_preset(str(local) if local else preset)
    if backbone.vocabulary_size != vocab_size:
        raise ValueError(
            f"{preset}: vocabulary {backbone.vocabulary_size} != tokenizer {vocab_size}"
        )
    return backbone


def run_encoder(
    name: str,
    backbone: keras.Model,
    encode: Encode,
    specials: Specials,
    cfg: dict[str, Any],
    corpus_dir: str | Path,
    out_dir: str | Path,
    seed: int,
    sources: Sequence[str] = ("synthetic", "synthetic_hn", "ami"),
) -> dict[str, Any]:
    """Fine-tune ``backbone`` on train, then write ``<out_dir>/<name>/<split>.parquet`` logits
    for every split. Returns the training history and counts."""
    from src.model1_notes.data import read_corpus_split

    def pack(split: str) -> Packed:
        meetings = read_corpus_split(corpus_dir, split, "sentence_text", sources)
        return pack_meetings(
            meetings,
            encode,
            specials,
            cfg["max_tokens"],
            cfg["max_sentence_tokens"],
            cfg["max_sentences"],
        )

    packed = {split: pack(split) for split in ("train", "validation", "test")}
    model = build_sentence_classifier(backbone, cfg["max_tokens"], cfg["max_sentences"])
    history = fine_tune(model, packed["train"], packed["validation"], cfg, seed)
    counts = {}
    for split, p in packed.items():
        rows = predict_logits(model, p, cfg["batch_size"])
        write_logits(Path(out_dir) / name / f"{split}.parquet", rows)
        counts[split] = len(rows)
    return {
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "sentences": counts,
    }
