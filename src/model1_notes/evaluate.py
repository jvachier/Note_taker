"""Model 1 v3 results table (PLAN.md 10.6-10.7).

Usage::

    uv run python -m src.model1_notes.evaluate --config configs/model1_v3_eval.yaml [--latency]

Every system is scored on the same test sentences, separately on the synthetic layer and on
**AMI test** (the headline), against the gold labels:

| System | How it selects | Zones |
|---|---|---|
| Lead-k | first ``fraction`` of each meeting, tuned on validation | - |
| Keyword rule | trigger regexes (config ``baselines``) | from the trigger |
| Qwen3-0.6B | soft score >= threshold tuned on validation | from its ``reason`` |
| RoBERTa-Large / DistilBERT | sigmoid(logit) >= threshold tuned on validation | - |
| Students | probability >= the seed's validation threshold | predicted |

Students are reported as mean +- std over seeds (their test predictions and ``metrics.json``
from :mod:`train`). On AMI test the kept sentences are also compared with the human summary:
ROUGE-1/2/L of all kept sentences vs the whole summary, ROUGE-L of predicted decisions vs the
*decisions* section and of action items vs the *actions* section. ``--latency`` times each
student on this machine's CPU for meetings of the configured sizes.

Writes ``<tables_dir>/model1_v3.md`` and ``<results_dir>/model1_v3_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.config import load_config
from src.eval.summary import rouge
from src.model1_notes.baselines import KeywordRule, lead_fraction, tune_lead
from src.model1_notes.metrics import ZONES, selection_metrics, tune_threshold

log = logging.getLogger(__name__)
LAYERS = ("synthetic", "ami")
Key = tuple[str, int]
SECTIONS = ("summary_abstract", "summary_decisions", "summary_actions", "summary_problems")


def read_layer(corpus_dir: str | Path, layer: str, split: str, table: str) -> list[dict[str, Any]]:
    """Rows of ``data/<layer>/<table>/<split>.parquet`` (empty if the file is missing)."""
    path = Path(corpus_dir) / "data" / layer / table / f"{split}.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else []


class Gold:
    """Gold sentences of one split, sorted by meeting and sentence, with lookups."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = sorted(rows, key=lambda r: (r["note_id"], r["sentence_id"]))
        self.keys: list[Key] = [(r["note_id"], r["sentence_id"]) for r in self.rows]
        self.importance = np.array([r["importance"] for r in self.rows], dtype=bool)
        self.zone = np.array([ZONES.index(r["zone"]) for r in self.rows], dtype=np.int64)
        self.hard_negative = np.array([r["hard_negative"] for r in self.rows], dtype=bool)
        self.layer = np.array(["ami" if r["source"] == "ami" else "synthetic" for r in self.rows])
        self.meetings: dict[str, list[int]] = {}
        for i, (note_id, _) in enumerate(self.keys):
            self.meetings.setdefault(note_id, []).append(i)

    @classmethod
    def load(cls, corpus_dir: str | Path, split: str) -> Gold:
        return cls(
            [r for layer in LAYERS for r in read_layer(corpus_dir, layer, split, "sentences")]
        )


class Prediction:
    """One system's test output aligned to :class:`Gold`."""

    def __init__(
        self,
        gold: Gold,
        picked: dict[Key, int],
        prob: dict[Key, float] | None = None,
        zone: dict[Key, str] | None = None,
    ) -> None:
        self.kept = np.array([picked.get(k, 0) for k in gold.keys], dtype=bool)
        self.prob = None if prob is None else np.array([prob.get(k, 0.0) for k in gold.keys])
        self.zone = None
        if zone is not None:
            names = [zone.get(k, "discussion") for k in gold.keys]
            self.zone = np.array([ZONES.index(z) for z in names], dtype=np.int64)


def score(gold: Gold, pred: Prediction) -> dict[str, Any]:
    """:func:`selection_metrics` per layer."""
    out = {}
    for layer in LAYERS:
        rows = gold.layer == layer
        if rows.any():
            out[layer] = selection_metrics(
                gold.importance[rows],
                gold.zone[rows],
                pred.kept[rows],
                None if pred.prob is None else pred.prob[rows],
                None if pred.zone is None else pred.zone[rows],
                gold.hard_negative[rows],
            )
    return out


def ami_rouge(
    gold: Gold, notes: dict[str, dict[str, Any]], pred: Prediction, use_stemmer: bool
) -> dict[str, float] | None:
    """ROUGE of the kept AMI sentences against the human summary (see module docstring)."""
    ids = [m for m in gold.meetings if notes.get(m, {}).get("summary_abstract")]
    if not ids:
        return None

    def text(note_id: str, zone: int | None = None) -> str:
        rows = gold.meetings[note_id]
        keep = [i for i in rows if pred.kept[i] and (zone is None or pred.zone[i] == zone)]
        return " ".join(gold.rows[i]["sentence_text"] for i in keep)

    full = [[" ".join(notes[m][s] or "" for s in SECTIONS)] for m in ids]
    out = {
        f"notes_{k}": round(v, 4)
        for k, v in rouge([text(m) for m in ids], full, use_stemmer).items()
    }
    if pred.zone is not None:
        for name, zone in (("decisions", "decision"), ("actions", "action_item")):
            refs = [
                (m, notes[m][f"summary_{name}"]) for m in ids if notes[m].get(f"summary_{name}")
            ]
            if refs:
                preds = [text(m, ZONES.index(zone)) for m, _ in refs]
                result = rouge(preds, [[r] for _, r in refs], use_stemmer)
                out[f"{name}_rougeL"] = round(result["rougeL"], 4)
    return out


def tuned_pick(
    val: Gold, val_prob: dict[Key, float], test_prob: dict[Key, float]
) -> tuple[float, dict[Key, int]]:
    """Threshold tuned on validation importance F1, applied to the test probabilities."""
    p = np.array([val_prob.get(k, 0.0) for k in val.keys])
    threshold, _ = tune_threshold(val.importance, p)
    return threshold, {k: int(v >= threshold) for k, v in test_prob.items()}


def read_teacher(teacher_dir: Path, name: str, split: str) -> dict[Key, dict[str, Any]] | None:
    """A teacher table keyed by ``(note_id, sentence_id)``, or None if it was not exported."""
    path = teacher_dir / name / f"{split}.parquet"
    if not path.exists():
        return None
    return {(r["note_id"], r["sentence_id"]): r for r in pq.read_table(path).to_pylist()}


def baseline_predictions(
    val: Gold, test: Gold, cfg: dict[str, Any]
) -> dict[str, tuple[Prediction, dict[str, Any]]]:
    """Lead-k (fraction tuned on validation) and the keyword rule."""
    bl = cfg["baselines"]
    fraction, _ = tune_lead(
        [val.importance[rows].astype(int).tolist() for rows in val.meetings.values()],
        bl["lead_fractions"],
    )
    lead: dict[Key, int] = {}
    for rows in test.meetings.values():
        for i, keep in zip(rows, lead_fraction(len(rows), fraction), strict=True):
            lead[test.keys[i]] = keep
    rule = KeywordRule(bl["keywords"])
    zones = {k: rule.zone(r["sentence_text"]) for k, r in zip(test.keys, test.rows, strict=True)}
    keyword = {k: int(z != "discussion") for k, z in zones.items()}
    return {
        "Lead-k": (Prediction(test, lead), {"fraction": fraction}),
        "Keyword rule": (Prediction(test, keyword, zone=zones), {}),
    }


def teacher_predictions(
    val: Gold, test: Gold, cfg: dict[str, Any]
) -> dict[str, tuple[Prediction, dict[str, Any]]]:
    """Qwen (soft score, zone from its reason) and the fine-tuned encoders (sigmoid logit)."""
    out = {}
    teacher_dir = Path(cfg["paths"]["teacher_dir"])
    for name, info in cfg["teachers"].items():
        val_t, test_t = (read_teacher(teacher_dir, name, s) for s in ("validation", "test"))
        if test_t is None or val_t is None:
            log.warning("teacher %s: no exported validation/test table, skipped", name)
            continue
        if name == "qwen":
            val_p = {k: r["score"] for k, r in val_t.items()}
            test_p = {k: r["score"] for k, r in test_t.items()}
            reason_zone = cfg["qwen_zone_of_reason"]
            zones = {k: reason_zone.get(r["reason"], "discussion") for k, r in test_t.items()}
        else:
            val_p = {k: 1 / (1 + math.exp(-r["logit"])) for k, r in val_t.items()}
            test_p = {k: 1 / (1 + math.exp(-r["logit"])) for k, r in test_t.items()}
            zones = None
        threshold, picked = tuned_pick(val, val_p, test_p)
        out[info["display"]] = (
            Prediction(test, picked, test_p, zones),
            {
                "threshold": threshold,
                "params": info["params"],
                "coverage": len(test_t) / len(test.keys),
            },
        )
    return out


def student_seed_predictions(
    test: Gold, results_dir: Path, name: str
) -> list[tuple[Prediction, dict[str, Any]]]:
    """Each complete seed's test prediction and its ``metrics.json``."""
    runs = []
    for seed_dir in sorted((results_dir / name).glob("seed_*")):
        metrics_path, preds_path = seed_dir / "metrics.json", seed_dir / "test_predictions.jsonl"
        if not (metrics_path.exists() and preds_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text())
        prob, zone = {}, {}
        for line in preds_path.read_text().splitlines():
            rec = json.loads(line)
            for sid, p, z in zip(rec["sentence_ids"], rec["importance"], rec["zone"], strict=True):
                prob[(rec["note_id"], sid)], zone[(rec["note_id"], sid)] = p, z
        picked = {k: int(p >= metrics["threshold"]) for k, p in prob.items()}
        runs.append((Prediction(test, picked, prob, zone), metrics))
    return runs


def measure_latency(model_path: Path, sentence_tokens: int, sizes: list[int], repeats: int) -> dict:
    """Median CPU milliseconds per meeting of ``n`` sentences (windows of ``max_sentences``)."""
    import keras

    import src.model1_notes.model  # noqa: F401  (registers the layers)
    from src.model1_notes.losses import NotesImportanceLoss  # noqa: F401

    model = keras.saving.load_model(model_path, compile=False)
    t, size = sentence_tokens, model.get_config()["max_sentences"]
    out = {}
    for n in sizes:
        windows = math.ceil(n / size)
        x = {
            "token_ids": np.full((windows, min(n, size), t), 5, dtype=np.int32),
            "speaker_ids": np.ones((windows, min(n, size)), dtype=np.int32),
        }
        model.predict_on_batch(x)  # compile
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict_on_batch(x)
            times.append(time.perf_counter() - start)
        out[str(n)] = round(1000 * float(np.median(times)), 1)
    return out


def mean_std(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def fmt_params(n: float | None) -> str:
    if n is None:
        return "–"
    n = float(n)
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


COLUMNS = [
    ("Synth imp F1", ("synthetic", "importance_f1")),
    ("Synth zone F1", ("synthetic", "zone_macro_f1")),
    ("Synth HN FP ↓", ("synthetic", "hard_negative_fp")),
    ("AMI imp F1", ("ami", "importance_f1")),
    ("AMI PR-AUC", ("ami", "pr_auc")),
    ("AMI dec F1", ("ami", "decision_f1")),
    ("AMI act F1", ("ami", "action_f1")),
    ("AMI R-1", ("rouge", "notes_rouge1")),
    ("AMI R-2", ("rouge", "notes_rouge2")),
    ("AMI R-L", ("rouge", "notes_rougeL")),
    ("AMI dec R-L", ("rouge", "decisions_rougeL")),
    ("Spans F1", ("spans", "span_f1")),
]


def row_values(entries: list[dict[str, Any]]) -> list[str]:
    """Column cells for one system (``entries``: one result dict per seed)."""
    cells = []
    for _, (group, key) in COLUMNS:
        cells.append(mean_std([(e.get(group) or {}).get(key) for e in entries]))
    return cells


def table(rows: list[tuple[str, str, list[dict[str, Any]], str]]) -> str:
    """Markdown table: system, params, metric columns, CPU latency."""
    head = ["System", "Params", *(c for c, _ in COLUMNS), "CPU ms (100 / 500 sent.)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, params, entries, latency in rows:
        lines.append("| " + " | ".join([name, params, *row_values(entries), latency]) + " |")
    return "\n".join(lines)


def evaluate(cfg: dict[str, Any], latency: bool = False) -> dict[str, Any]:
    """Score every available system; write the table and the JSON (see module docstring)."""
    corpus, results_dir = cfg["paths"]["corpus_dir"], Path(cfg["paths"]["results_dir"])
    val, test = Gold.load(corpus, "validation"), Gold.load(corpus, "test")
    notes = {r["note_id"]: r for r in read_layer(corpus, "ami", "test", "notes")}
    stem = cfg["rouge"]["use_stemmer"]
    systems = {**baseline_predictions(val, test, cfg), **teacher_predictions(val, test, cfg)}
    report: dict[str, Any] = {}
    rows = []
    for name, (pred, info) in systems.items():
        entry = {**score(test, pred), "rouge": ami_rouge(test, notes, pred, stem), **info}
        report[name] = entry
        rows.append((name, fmt_params(info.get("params")), [entry], "–"))
    for student in cfg["students"]:
        runs = student_seed_predictions(test, results_dir, student)
        if not runs:
            log.warning("student %s: no complete seed, skipped", student)
            continue
        entries = [
            {
                **score(test, pred),
                "rouge": ami_rouge(test, notes, pred, stem),
                "spans": m.get("test", {}).get("spans"),
                "seed": m["seed"],
            }
            for pred, m in runs
        ]
        lat = "–"
        if latency:
            seed = runs[0][1]["seed"]
            path = Path(cfg["paths"]["checkpoints_dir"]) / student / f"seed_{seed}" / "best.keras"
            if path.exists():
                lat_cfg = cfg["latency"]
                ms = measure_latency(
                    path, lat_cfg["sentence_tokens"], lat_cfg["sentences"], lat_cfg["repeats"]
                )
                lat = " / ".join(str(v) for v in ms.values())
                entries[0]["latency_ms"] = ms
        report[student] = {"seeds": entries}
        rows.append((student, fmt_params(runs[0][1]["params"]), entries, lat))
    md = "\n\n".join(
        [
            "# Model 1 v3: meeting notes (test, gold labels)",
            "Synthetic = `meeting-notes-extractive-v1` synthetic test; **AMI = real meetings, the "
            "headline**. Students: mean ± std over seeds. Teachers' rows use their cached test "
            "outputs (Notebook 1); thresholds are tuned on validation. HN FP = share of gold "
            "hard negatives kept (lower is better). Spans: owner / deadline exact-match F1 "
            "(synthetic test).",
            table(rows),
        ]
    )
    tables = Path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "model1_v3.md").write_text(md + "\n")
    (results_dir / "model1_v3_eval.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """Entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model1_v3_eval.yaml"))
    parser.add_argument("--latency", action="store_true", help="time students on this CPU")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    cfg["baselines"] = load_config(Path(args.config).parent / "model1_v3.yaml")["baselines"]
    evaluate(cfg, args.latency)
    print((Path(cfg["paths"]["tables_dir"]) / "model1_v3.md").read_text())


if __name__ == "__main__":
    main()
