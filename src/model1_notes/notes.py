"""Meeting notes from a transcript with a trained v3 student (PLAN.md 10.1).

Usage::

    uv run python -m src.model1_notes.notes --model checkpoints/model1_v3_E4/seed_0 \\
        --tokenizer data/tokenizer --input meeting.txt [--asr --model2 checkpoints/model2/seed_0]

The input has one ``Speaker: sentence`` per line. With ``--asr`` (lowercase, unpunctuated
speech) each sentence first goes through **Model 2** to restore punctuation and casing. The
student scores every sentence; sentences at or above the threshold (the seed's validation
threshold from ``metrics.json`` unless given) become notes, grouped by their zone (the
calibrated ``zone_thresholds`` of ``metrics.json`` when present, see
:mod:`~src.model1_notes.calibrate`):

    ## Decisions      zone = decision
    ## Action items   zone = action_item, with the predicted owner and deadline
    ## Key points     any other zone

Sentences keep their meeting order inside each section.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import keras
import numpy as np
from tokenizers import Tokenizer

import src.model1_notes.losses  # noqa: F401  (registers the custom objects for loading)
import src.model1_notes.model  # noqa: F401
import src.model2_cleanup.model  # noqa: F401
from src.config import load_config
from src.model1_notes.calibrate import assign_zones
from src.model1_notes.data import ZONES, encode_sentence, speaker_numbers
from src.model1_notes.metrics import span_entities
from src.model1_notes.postprocess import postprocess
from src.model1_notes.teacher.windows import best_window, window_starts
from src.tokenizer.tokenizer import load_tokenizer

SECTION_OF_ZONE = {"decision": "Decisions", "action_item": "Action items"}
SECTIONS = ("Decisions", "Action items", "Key points")


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
    followups: list[Note] = field(default_factory=list)  # replies attached by postprocess


def parse_transcript(text: str) -> tuple[list[str], list[str]]:
    """``Speaker: sentence`` lines -> speakers, sentences (lines without a colon: speaker "?")."""
    speakers, sentences = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        speaker, sep, sentence = line.partition(":")
        speakers.append(speaker.strip() if sep else "?")
        sentences.append(sentence.strip() if sep else line.strip())
    return speakers, sentences


def predict(
    model: keras.Model,
    tokenizer: Tokenizer,
    speakers: Sequence[str],
    sentences: Sequence[str],
    max_tokens: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[int]]:
    """Importance probability (N,), zone probabilities (N, 5), BIO span tags (per subword) and
    the sentence each one depends on (-1 = none) for every sentence; long meetings in windows,
    each sentence from its most central one."""
    size = model.get_config()["max_sentences"]
    n = len(sentences)
    numbers, _ = speaker_numbers(speakers)
    encoded = [encode_sentence(tokenizer, s, None, max_tokens)[0] for s in sentences]
    starts = window_starts(n, size, overlap)
    outputs = []
    for start in starts:
        rows = range(start, min(start + size, n))
        x = {
            "token_ids": np.stack([encoded[i] for i in rows])[None],
            "speaker_ids": np.array([[numbers[i] for i in rows]], dtype=np.int32),
        }
        outputs.append(model.predict_on_batch(x))
    prob, zone_prob, spans, depends = np.zeros(n), np.zeros((n, len(ZONES))), [], []
    for i, w in enumerate(best_window(n, size, overlap)):
        j = i - starts[w]
        out = outputs[w]
        prob[i] = 1 / (1 + np.exp(-float(np.asarray(out["importance"])[0, j])))
        logits = np.asarray(out["zone"], dtype=np.float64)[0, j]
        zone_prob[i] = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        spans.append(np.asarray(out["spans"])[0, j].argmax(-1).tolist())
        pointer = np.asarray(out["depends_on"])[0, j]
        k = int(pointer.argmax())
        depends.append(starts[w] + k if k < len(pointer) - 1 else -1)  # last column = none
    return prob, zone_prob, spans, depends


def annotate(
    model: keras.Model,
    tokenizer: Tokenizer,
    speakers: Sequence[str],
    sentences: Sequence[str],
    settings: dict,
) -> list[Note]:
    """Every sentence with its score, zone, kept flag and (action items) owner / deadline.

    ``settings``: ``max_sentence_tokens``, ``window_overlap``, ``threshold`` and optional
    ``zone_thresholds`` and ``postprocess`` (:mod:`~src.model1_notes.postprocess`) - the released
    model's ``config.json`` ``inference`` section.
    """
    if not sentences:
        return []
    max_tokens = settings["max_sentence_tokens"]
    overlap = min(settings["window_overlap"], model.get_config()["max_sentences"] - 1)
    prob, zone_prob, tags, depends = predict(
        model, tokenizer, speakers, sentences, max_tokens, overlap
    )
    zones = [ZONES[z] for z in assign_zones(zone_prob, settings.get("zone_thresholds"))]
    notes = []
    for i, (speaker, text) in enumerate(zip(speakers, sentences, strict=True)):
        kept = bool(prob[i] >= settings["threshold"])
        spans = span_text(tokenizer, text, tags[i], max_tokens)
        notes.append(
            Note(
                i,
                speaker,
                text,
                zones[i],
                float(prob[i]),
                kept,
                spans.get("owner"),
                spans.get("deadline"),
            )
        )
    return postprocess(notes, depends, settings.get("postprocess"))


def span_text(tokenizer: Tokenizer, sentence: str, tags: Sequence[int], max_tokens: int) -> dict:
    """``{"owner": text, "deadline": text}`` from the predicted BIO tags (first entity per role)."""
    enc = tokenizer.encode(sentence, add_special_tokens=False)
    offsets = enc.offsets[: max_tokens - 2]
    inner = list(tags[1 : len(offsets) + 1])  # position 0 is [CLS]
    found: dict[str, str] = {}
    for start, end, role in sorted(span_entities(inner)):
        if role not in found and end <= len(offsets):
            found[role] = sentence[offsets[start][0] : offsets[end - 1][1]]
    return found


def make_notes(annotated: Sequence[Note]) -> list[Note]:
    """The kept sentences, in meeting order."""
    return [n for n in annotated if n.kept]


def render(notes: Sequence[Note]) -> str:
    """Markdown notes: Decisions, Action items (owner / deadline), Key points."""
    sections: dict[str, list[str]] = {s: [] for s in SECTIONS}
    for note in notes:
        section = SECTION_OF_ZONE.get(note.zone, "Key points")
        line = f"- {note.text} ({note.speaker})"
        if section == "Action items" and (note.owner or note.deadline):
            line += f" — owner: {note.owner or '?'}, due: {note.deadline or '?'}"
        for reply in note.followups:
            line += f"\n  - ↳ {reply.speaker}: {reply.text}"
        sections[section].append(line)
    parts = []
    for name in SECTIONS:
        parts.append(f"## {name}\n" + ("\n".join(sections[name]) or "- (none)"))
    return "\n\n".join(parts) + "\n"


def clean_asr(
    sentences: Sequence[str],
    model2: keras.Model,
    tokenizer: Tokenizer,
    window_words: int,
    max_len: int,
) -> list[str]:
    """Restore punctuation and casing with Model 2 (ASR-style input)."""
    from src.model2_cleanup.infer import clean_text

    return [clean_text(model2, tokenizer, s, window_words, max_len) or s for s in sentences]


def notes_from_transcript(
    text: str,
    model_dir: Path,
    tokenizer: Tokenizer,
    cfg: dict,
    threshold: float | None = None,
    cleanup: tuple[keras.Model, dict] | None = None,
) -> str:
    """The whole pipeline (see module docstring); returns Markdown.

    ``cfg`` is the v3 config (sentence length, window overlap); ``cleanup`` is Model 2 and its
    ``data`` config section, for ASR-style input.
    """
    speakers, sentences = parse_transcript(text)
    if not sentences:
        return render([])
    if cleanup is not None:
        model2, m2_data = cleanup
        sentences = clean_asr(
            sentences, model2, tokenizer, m2_data["window_words"], m2_data["max_len"]
        )
    model = keras.saving.load_model(model_dir / "best.keras", compile=False)
    metrics = model_dir / "metrics.json"
    saved = json.loads(metrics.read_text()) if metrics.exists() else {}
    settings = {
        "max_sentence_tokens": cfg["data"]["max_sentence_tokens"],
        "window_overlap": cfg["data"]["window_overlap"],
        "threshold": saved.get("threshold", 0.5) if threshold is None else threshold,
        "zone_thresholds": saved.get("zone_thresholds"),
        "postprocess": cfg["inference"].get("postprocess"),
    }
    return render(make_notes(annotate(model, tokenizer, speakers, sentences, settings)))


def main(argv: list[str] | None = None) -> None:
    """Entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="dir with best.keras")
    parser.add_argument("--config", type=Path, default=Path("configs/model1_v3.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--asr", action="store_true", help="clean the text with Model 2 first")
    parser.add_argument("--model2", type=Path, default=Path("checkpoints/model2/seed_0"))
    parser.add_argument("--model2-config", type=Path, default=Path("configs/model2.yaml"))
    args = parser.parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer)
    cleanup = None
    if args.asr:
        model2 = keras.saving.load_model(args.model2 / "best.keras", compile=False)
        cleanup = (model2, load_config(args.model2_config)["data"])
    cfg = load_config(args.config)
    print(
        notes_from_transcript(
            args.input.read_text(), args.model, tokenizer, cfg, args.threshold, cleanup
        )
    )


if __name__ == "__main__":
    main()
