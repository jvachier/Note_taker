"""Self-contained code to load and run meeting-notes-en-21m (extractive meeting notes).

Only needs ``keras>=3.6``, ``jax`` and ``tokenizers``. Importing this module registers the
model's custom layers (package ``dialogue_models``) so ``model.keras`` loads::

    import modeling
    notetaker = modeling.NoteTaker.from_folder(".")   # the downloaded repo folder
    print(notetaker("Anna: Let's launch on the 15th.\\nBen: Okay, I'll send an update today."))

The model reads a meeting as a list of sentences with their speakers. For every sentence it
predicts an importance score, a zone (context / discussion / decision / action_item / closing)
and owner / deadline spans. Sentences above the threshold become notes, grouped into
Decisions, Action items and Key points. A few rules (``config.json`` ``postprocess``) attach
short replies ("OK, I will email him tonight.") under the request they answer and give
first-person commitments their speaker as owner. The model selects sentences; it never writes
new text.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import ops  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PAD_ID, CLS_ID, SEP_ID = 0, 2, 3
NUM_SPEAKERS = 8
NEG = -1e9
ZONES = ("context", "discussion", "decision", "action_item", "closing")
CALIBRATED = ("decision", "action_item")  # zones with their own threshold (config.json)
SPAN_TAGS = ("O", "B-owner", "I-owner", "B-deadline", "I-deadline")
SECTION_OF_ZONE = {"decision": "Decisions", "action_item": "Action items"}
SECTIONS = ("Decisions", "Action items", "Key points")
ATTACH_TO = ("action_item", "decision")  # zones a reply can be attached to

# ----------------------------------------------------------------------------- layers


@keras.saving.register_keras_serializable(package="dialogue_models")
class TokenAndPositionEmbedding(keras.layers.Layer):
    """Sum of token, learned position and (optionally) speaker embeddings, then dropout."""

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
        self.vocab_size, self.max_len, self.d_model = vocab_size, max_len, d_model
        self.num_speakers, self.dropout_rate, self.init_std = num_speakers, dropout, init_std
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

    def build(self, input_shape):
        self.token_embedding.build(input_shape)
        if self.speaker_embedding is not None:
            self.speaker_embedding.build(input_shape)
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.max_len, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=self.init_std),
        )
        super().build(input_shape)

    def call(self, token_ids, speaker_ids=None, training=None):
        length = ops.shape(token_ids)[-1]
        x = self.token_embedding(token_ids)
        x = x + ops.cast(self.position_embedding[:length], x.dtype)[None, :, :]
        if self.speaker_embedding is not None:
            x = x + self.speaker_embedding(speaker_ids)
        return self.dropout(x, training=training)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.d_model)

    def get_config(self):
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


@keras.saving.register_keras_serializable(package="dialogue_models")
class TransformerEncoderBlock(keras.layers.Layer):
    """Pre-LayerNorm block: ``x + MHA(LN(x))`` then ``x + FFN(LN(x))``."""

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
        self.d_model, self.num_heads, self.ffn_dim = d_model, num_heads, ffn_dim
        self.dropout_rate, self.init_std = dropout, init_std
        init = keras.initializers.TruncatedNormal(stddev=init_std)
        self.attn_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="attn_norm")
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            kernel_initializer=init,
            name="attention",
        )
        self.attn_dropout = keras.layers.Dropout(dropout)
        self.ffn_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="ffn_norm")
        self.ffn_in = keras.layers.Dense(
            ffn_dim, activation="gelu", kernel_initializer=init, name="ffn_in"
        )
        self.ffn_out = keras.layers.Dense(d_model, kernel_initializer=init, name="ffn_out")
        self.ffn_dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        self.attn_norm.build(input_shape)
        self.attention.build(input_shape, input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn_in.build(input_shape)
        self.ffn_out.build((*input_shape[:-1], self.ffn_dim))
        super().build(input_shape)

    def call(self, x, padding_mask=None, training=None):
        mask = None
        if padding_mask is not None:
            mask = ops.logical_and(padding_mask[:, :, None], padding_mask[:, None, :])
        h = self.attn_norm(x)
        h = self.attention(h, h, attention_mask=mask, training=training)
        x = x + self.attn_dropout(h, training=training)
        h = self.ffn_out(self.ffn_in(self.ffn_norm(x)))
        return x + self.ffn_dropout(h, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
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
    """Embeddings, ``num_layers`` pre-LN blocks and a final LayerNorm; token 0 is padding."""

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
        self.vocab_size, self.max_len, self.d_model = vocab_size, max_len, d_model
        self.num_layers, self.num_heads, self.ffn_dim = num_layers, num_heads, ffn_dim
        self.dropout_rate, self.init_std = dropout, init_std
        self.num_speakers, self.pad_id = num_speakers, pad_id
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

    def build(self, input_shape):
        self.embedding.build(input_shape)
        hidden = (*input_shape, self.d_model)
        for block in self.blocks:
            block.build(hidden)
        self.final_norm.build(hidden)
        super().build(input_shape)

    def call(self, token_ids, speaker_ids=None, training=None):
        padding_mask = ops.not_equal(token_ids, self.pad_id)
        x = self.embedding(token_ids, speaker_ids, training=training)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, training=training)
        return self.final_norm(x)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.d_model)

    def get_config(self):
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


@keras.saving.register_keras_serializable(package="dialogue_models")
class DependsOnPointer(keras.layers.Layer):
    """Pointer logits from each sentence to every earlier real sentence, plus a "none" column."""

    def __init__(self, d_model: int, init_std: float = 0.02, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.d_model, self.init_std = d_model, init_std
        init = keras.initializers.TruncatedNormal(stddev=init_std)
        self.query = keras.layers.Dense(d_model, kernel_initializer=init, name="query")
        self.key = keras.layers.Dense(d_model, kernel_initializer=init, name="key")
        self.none = keras.layers.Dense(1, kernel_initializer=init, name="none")

    def build(self, input_shape):
        for layer in (self.query, self.key, self.none):
            layer.build(input_shape)
        super().build(input_shape)

    def call(self, hidden, sentence_mask):
        h = ops.cast(hidden, "float32")
        scores = ops.matmul(self.query(h), ops.swapaxes(self.key(h), 1, 2))
        scores = scores / ops.sqrt(ops.cast(self.d_model, "float32"))
        s = ops.shape(h)[1]
        earlier = ops.arange(s)[None, :] < ops.arange(s)[:, None]
        allowed = ops.logical_and(earlier[None, :, :], sentence_mask[:, None, :])
        scores = ops.where(allowed, scores, NEG)
        return ops.concatenate([scores, self.none(h)], axis=-1)

    def compute_output_shape(self, hidden_shape, sentence_mask_shape=None):
        s = hidden_shape[1]
        return (hidden_shape[0], s, None if s is None else s + 1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "init_std": self.init_std}


@keras.saving.register_keras_serializable(package="dialogue_models")
class NotesModel(keras.Model):
    """Hierarchical student: a sentence encoder, a meeting encoder over the sentences' ``[CLS]``
    states (+ sentence position, speaker and speaker-change embeddings) and per-sentence heads.

    Inputs: ``token_ids`` (batch, S, T) and ``speaker_ids`` (batch, S). Outputs: ``importance``
    (batch, S), ``zone`` (batch, S, 5), ``reason`` (batch, S, 9), ``depends_on``
    (batch, S, S + 1) and ``spans`` (batch, S, T, 5) logits.
    """

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

    def call(self, inputs, training=None):
        token_ids = ops.cast(inputs["token_ids"], "int32")
        speakers = ops.cast(inputs["speaker_ids"], "int32")
        b, s, t = ops.shape(token_ids)
        real_tokens = ops.not_equal(token_ids, PAD_ID)
        token_speakers = ops.where(real_tokens, speakers[:, :, None], 0)
        hidden = self.encoder(
            ops.reshape(token_ids, (-1, t)), ops.reshape(token_speakers, (-1, t)), training=training
        )
        tokens = ops.reshape(hidden, (b, s, t, self.args["d_model"]))
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

    def get_config(self):
        return {**super().get_config(), **self.args}


# ----------------------------------------------------------------------------- text

_SPEAKER = re.compile(r"^\s*([^:]{1,40}?)\s*:\s*(.+)$")
# A new sentence starts with a capital, digit or quote after ". ", "! " or "? ".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def parse_transcript(text: str) -> tuple[list[str], list[str]]:
    """``Speaker: text`` lines -> the speaker and text of every sentence, in order.

    A line without a ``name:`` prefix continues the previous speaker's turn.
    """
    speakers: list[str] = []
    sentences: list[str] = []
    current = "Speaker 1"
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _SPEAKER.match(line)
        if match:
            current, turn = match.group(1).strip("# "), match.group(2).strip()
        else:
            turn = line.strip()
        for sentence in split_sentences(turn) or [turn]:
            speakers.append(current)
            sentences.append(sentence)
    return speakers, sentences


def split_sentences(text: str) -> list[str]:
    """Split a turn into sentences after ``.``, ``!`` or ``?``."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def speaker_numbers(speakers: Sequence[str]) -> list[int]:
    """1-based speaker numbers by first appearance (speakers after the 8th share number 8)."""
    order: dict[str, int] = {}
    for s in speakers:
        order.setdefault(s, len(order) + 1)
    return [min(order[s], NUM_SPEAKERS) for s in speakers]


def window_starts(length: int, size: int, overlap: int) -> list[int]:
    """Start of each window of ``size`` sentences overlapping by ``overlap``."""
    if length <= size:
        return [0]
    return [*range(0, length - size, size - overlap), length - size]


def best_window(length: int, size: int, overlap: int) -> list[int]:
    """For each sentence, the window where it is farthest from both edges."""
    starts = window_starts(length, size, overlap)
    best = []
    for i in range(length):
        margins = [min(i - s, s + size - 1 - i) if s <= i < s + size else -1 for s in starts]
        best.append(max(range(len(starts)), key=lambda w: (margins[w], -w)))
    return best


def span_entities(tags: Sequence[int]) -> set[tuple[int, int, str]]:
    """``(start, end, role)`` entities of one BIO sequence (an ``I`` without ``B`` starts one)."""
    entities, start, role = set(), None, None
    for i, t in enumerate([*tags, 0]):
        name = SPAN_TAGS[t] if 0 <= t < len(SPAN_TAGS) else "O"
        inside = name.startswith("I-") and role == name[2:]
        if start is not None and not inside:
            entities.add((start, i, role))
            start, role = None, None
        if name != "O" and not inside:
            start, role = i, name[2:]
    return entities


# ----------------------------------------------------------------------------- inference


@dataclass
class Note:
    """One sentence with the model's verdict (``kept``: it belongs in the notes)."""

    index: int
    speaker: str
    text: str
    zone: str
    score: float
    kept: bool = True
    owner: str | None = None
    deadline: str | None = None
    followups: list[Note] = field(default_factory=list)  # replies attached under it


def assign_zones(zone_prob: np.ndarray, thresholds: dict[str, float] | None) -> np.ndarray:
    """Zone index per sentence: argmax, overridden by a calibrated zone above its threshold."""
    zones = zone_prob.argmax(-1)
    if not thresholds:
        return zones
    ratios = np.stack([zone_prob[:, ZONES.index(z)] / thresholds[z] for z in CALIBRATED], -1)
    calibrated = np.array([ZONES.index(z) for z in CALIBRATED])[ratios.argmax(-1)]
    return np.where(ratios.max(-1) >= 1.0, calibrated, zones)


def postprocess(notes: list[Note], depends: Sequence[int], cfg: dict[str, Any] | None) -> None:
    """Rules applied after the model (no change to which sentences are kept).

    1. A kept first-person action item ("I'll ...") without a predicted owner gets its speaker.
    2. An unkept short reply ("Yes, I will book them today.", "OK, will do.") from another
       speaker is attached under the kept request or decision it answers: the sentence the
       ``depends_on`` head points to (if ``use_pointer``), else the nearest kept sentence among
       the ``lookback`` previous ones. A first-person reply gives an action item its missing
       owner and deadline.
    """
    if not cfg:
        return

    def matcher(patterns: Sequence[str]) -> Callable[[str], bool]:
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        return lambda text: any(p.search(text) for p in compiled)

    is_commitment = matcher(cfg["commitment_patterns"])
    is_confirmation = matcher(cfg["confirmation_patterns"])
    for note in notes:
        if note.kept and note.zone == "action_item" and not note.owner and is_commitment(note.text):
            note.owner = note.speaker
    attached: dict[int, int] = {}  # reply index -> request index
    for note in notes:
        if note.kept or len(note.text.split()) > cfg["confirmation_max_words"]:
            continue
        confirms = is_confirmation(note.text)
        if not (confirms or is_commitment(note.text)):
            continue
        pointer = depends[note.index] if cfg.get("use_pointer", True) else -1
        target = pointer if 0 <= pointer < note.index else -1
        if target < 0:
            back = range(note.index - 1, max(note.index - 1 - cfg["lookback"], -1), -1)
            target = next((j for j in back if notes[j].kept or j in attached), -1)
        if target < 0 or (target in attached and not confirms):
            continue  # a bare commitment ("I'll ...") does not join another reply's chain
        request = notes[attached.get(target, target)]
        if not request.kept or request.zone not in ATTACH_TO or request.speaker == note.speaker:
            continue
        request.followups.append(note)
        attached[note.index] = request.index
        if request.zone == "action_item" and is_commitment(note.text):
            request.owner = request.owner or note.speaker
            request.deadline = request.deadline or note.deadline


def render(notes: Sequence[Note]) -> str:
    """Markdown notes from the kept sentences: Decisions, Action items, Key points."""
    sections: dict[str, list[str]] = {s: [] for s in SECTIONS}
    for note in notes:
        if not note.kept:
            continue
        section = SECTION_OF_ZONE.get(note.zone, "Key points")
        line = f"- {note.text} ({note.speaker})"
        if section == "Action items" and (note.owner or note.deadline):
            line += f" — owner: {note.owner or '?'}, due: {note.deadline or '?'}"
        for reply in note.followups:
            line += f"\n  - ↳ {reply.speaker}: {reply.text}"
        sections[section].append(line)
    parts = [f"## {name}\n" + ("\n".join(sections[name]) or "- (none)") for name in SECTIONS]
    return "\n\n".join(parts) + "\n"


class NoteTaker:
    """Extractive meeting notes from a transcript (one ``Speaker: text`` line per turn).

    Meetings longer than the model's window (256 sentences) are read in overlapping windows;
    every sentence takes its prediction from the window where it is farthest from the edges.
    """

    def __init__(self, model: keras.Model, tokenizer: Tokenizer, settings: dict[str, Any]):
        self.model, self.tokenizer, self.settings = model, tokenizer, settings

    @classmethod
    def from_folder(cls, folder: str | Path) -> NoteTaker:
        """Load ``model.keras``, ``tokenizer.json`` and ``config.json`` from a folder."""
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        model = keras.saving.load_model(folder / "model.keras", compile=False)
        tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        return cls(model, tokenizer, config["inference"])

    def _encode(self, sentence: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        max_tokens = self.settings["max_sentence_tokens"]
        enc = self.tokenizer.encode(sentence, add_special_tokens=False)
        ids = enc.ids[: max_tokens - 2]
        token_ids = np.full(max_tokens, PAD_ID, dtype=np.int32)
        token_ids[: len(ids) + 2] = [CLS_ID, *ids, SEP_ID]
        return token_ids, enc.offsets[: len(ids)]

    def annotate(
        self, speakers: Sequence[str], sentences: Sequence[str], threshold: float | None = None
    ) -> list[Note]:
        """Every sentence with its score, zone, kept flag, owner / deadline and attached replies.

        ``threshold`` overrides the released keep threshold (lower keeps more sentences).
        """
        n = len(sentences)
        if n == 0:
            return []
        size = self.model.get_config()["max_sentences"]
        overlap = min(self.settings["window_overlap"], size - 1)
        threshold = self.settings["threshold"] if threshold is None else threshold
        numbers = speaker_numbers(speakers)
        encoded = [self._encode(s) for s in sentences]
        starts = window_starts(n, size, overlap)
        outputs = []
        for start in starts:
            rows = range(start, min(start + size, n))
            x = {
                "token_ids": np.stack([encoded[i][0] for i in rows])[None],
                "speaker_ids": np.array([[numbers[i] for i in rows]], dtype=np.int32),
            }
            outputs.append({k: np.asarray(v) for k, v in self.model.predict_on_batch(x).items()})
        prob, zone_prob = np.zeros(n), np.zeros((n, len(ZONES)))
        tags, depends = [], []
        for i, w in enumerate(best_window(n, size, overlap)):
            j, out = i - starts[w], outputs[w]
            prob[i] = 1 / (1 + np.exp(-float(out["importance"][0, j])))
            logits = out["zone"][0, j].astype(np.float64)
            zone_prob[i] = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
            tags.append(out["spans"][0, j].argmax(-1).tolist())
            pointer = out["depends_on"][0, j]
            k = int(pointer.argmax())
            depends.append(starts[w] + k if k < len(pointer) - 1 else -1)  # last column = none
        zones = assign_zones(zone_prob, self.settings.get("zone_thresholds"))
        notes = []
        for i, (speaker, text) in enumerate(zip(speakers, sentences, strict=True)):
            spans = self._spans(text, encoded[i][1], tags[i])
            notes.append(
                Note(
                    i,
                    speaker,
                    text,
                    ZONES[zones[i]],
                    float(prob[i]),
                    bool(prob[i] >= threshold),
                    spans.get("owner"),
                    spans.get("deadline"),
                )
            )
        postprocess(notes, depends, self.settings.get("postprocess"))
        return notes

    @staticmethod
    def _spans(sentence: str, offsets: Sequence[tuple[int, int]], tags: Sequence[int]) -> dict:
        """``{"owner": text, "deadline": text}`` from the BIO tags (first entity per role)."""
        inner = list(tags[1 : len(offsets) + 1])  # position 0 is [CLS]
        found: dict[str, str] = {}
        for start, end, role in sorted(span_entities(inner)):
            if role not in found and end <= len(offsets):
                found[role] = sentence[offsets[start][0] : offsets[end - 1][1]]
        return found

    def __call__(
        self,
        transcript: str,
        clean: Callable[[str], str] | None = None,
        threshold: float | None = None,
    ) -> str:
        """Markdown notes for ``transcript``.

        ``clean`` restores punctuation and casing of each turn first (e.g. the ``Cleaner`` of
        ``<hf-user>/punctuation-casing-en-7m``); use it for lowercase, unpunctuated transcripts.
        """
        if clean is not None:
            lines = []
            for line in transcript.splitlines():
                match = _SPEAKER.match(line)
                if match:
                    lines.append(f"{match.group(1)}: {clean(match.group(2)) or match.group(2)}")
                elif line.strip():
                    lines.append(clean(line) or line)
            transcript = "\n".join(lines)
        return render(self.annotate(*parse_transcript(transcript), threshold=threshold))
