"""Load the saved results of both models for the figures and the results report.

Reads only what training and evaluation already wrote: per-seed ``metrics.json`` and
``history.csv`` under ``results/``, the Model 1 test predictions, the teacher and oracle labels
and DialogSum test. No model is loaded. Model 1 system metrics come from
``results/model1_eval/*.json`` when :mod:`src.model1_scorer.evaluate` has run (these include
BERTScore); otherwise ROUGE and agreement are recomputed with the same functions, without
BERTScore.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.io import read_jsonl
from src.eval.metrics import spearman
from src.eval.summary import ROUGE_TYPES
from src.model1_scorer.evaluate import (
    Run,
    evaluate_run,
    labels_run,
    lead_run,
    oracle_run,
    selection,
)

log = logging.getLogger(__name__)

DATASET_NAMES = {"wikitext": "WikiText-103 test", "dialogsum": "DialogSum test"}
MODEL2_CLASSES = [
    ("Comma", ("punct", "per_class", "comma", "f1")),
    ("Period", ("punct", "per_class", "period", "f1")),
    ("Question", ("punct", "per_class", "question", "f1")),
    ("Capitalised", ("casing", "per_class", "capitalised", "f1")),
    ("UPPER", ("casing", "per_class", "upper", "f1")),
]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample std (0 for a single value)."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def read_history(path: Path) -> dict[str, np.ndarray]:
    """``history.csv`` as ``{column: values}``."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def seed_histories(exp_dir: Path) -> list[dict[str, np.ndarray]]:
    """One history per finished seed of an experiment."""
    return [read_history(p) for p in sorted(exp_dir.glob("seed_*/history.csv"))]


def epoch_stats(histories: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """``(epochs, 2)`` mean and std over seeds per epoch; seeds may stop at different epochs."""
    n = max(len(h[key]) for h in histories)
    return np.array([mean_std([h[key][e] for h in histories if e < len(h[key])]) for e in range(n)])


# ----------------------------------------------------------------------------- Model 2


def get_path(d: dict[str, Any], path: Sequence[str]) -> float:
    """``d[path[0]][path[1]]...`` as a float."""
    for key in path:
        d = d[key]
    return float(d)


def model2_test_sets(seeds: dict[int, dict[str, Any]]) -> dict[str, list[tuple[float, str]]]:
    """``{dataset: [(noise, test set key), ...]}`` sorted by noise."""
    out: dict[str, list[tuple[float, str]]] = {}
    for key, entry in next(iter(seeds.values()))["test"].items():
        out.setdefault(key.split("_")[1], []).append((entry["noise"], key))
    return {d: sorted(v) for d, v in out.items()}


def model2_value(
    seeds: dict[int, dict[str, Any]], key: str, system: str, path: Sequence[str]
) -> tuple[float, float]:
    """Mean and std over seeds of one Model 2 test metric (``system``: model or baseline)."""
    return mean_std([get_path(m["test"][key][system], path) for m in seeds.values()])


# ----------------------------------------------------------------------------- Model 1


@dataclass
class Model1Results:
    """Every Model 1 system on the shared test dialogues.

    Attributes:
        dialogues: DialogSum test dialogues by ID.
        ids: dialogues labelled by every teacher (the evaluation set).
        runs: system name -> one run per seed (one run for lead-3, oracle and teachers).
        metrics: system name -> metrics per run (as written by :mod:`evaluate`).
        params: system name -> parameter count (``None`` for lead-3 and the oracle).
        reference: name of the reference teacher system.
        students: student system names.
        top_k: utterances selected per dialogue.
    """

    dialogues: dict[str, dict]
    ids: list[str]
    runs: dict[str, list[Run]]
    metrics: dict[str, list[dict[str, Any]]]
    params: dict[str, float | None]
    reference: str
    students: list[str]
    top_k: int

    def value(self, name: str, fn: Callable[[dict[str, Any]], float]) -> tuple[float, float]:
        """Mean and std over runs (seeds) of one metric of a system."""
        return mean_std([fn(m) for m in self.metrics[name]])


def display_name(name: str) -> str:
    """Readable system name: ``model1_student_D_noMLM`` -> ``Student D (no MLM)``."""
    if name.startswith("Teacher "):
        return "Teacher " + name.split(" ", 1)[1].replace("qwen", "").replace("_", ".").upper()
    if name.startswith("model1_student_"):
        rest = name.removeprefix("model1_student_")
        return "Student " + rest.replace("_noMLM", " (no MLM)")
    return name


def group(name: str, students: Sequence[str]) -> str:
    """``reference``, ``teacher`` or ``student``."""
    if name in students:
        return "student"
    return "teacher" if name.startswith("Teacher ") else "reference"


def parse_params(text: str) -> float:
    """``"7B"`` -> 7e9, ``"1.5B"`` -> 1.5e9, ``"17M"`` -> 1.7e7."""
    scale = {"B": 1e9, "M": 1e6, "K": 1e3}
    return float(text[:-1]) * scale[text[-1]] if text[-1] in scale else float(text)


def load_model1(eval_cfg: dict[str, Any]) -> Model1Results:
    """Build every system's runs as :mod:`evaluate` does, and load or recompute its metrics."""
    paths = eval_cfg["paths"]
    labels, results = Path(paths["labels_dir"]), Path(paths["results_dir"])
    dialogues = {
        r["id"]: r for r in read_jsonl(Path(paths["raw_dir"]) / "dialogsum" / "test.jsonl")
    }
    teachers = {t: labels_run(labels / t / "test.jsonl") for t in eval_cfg["teachers"]}
    ids = sorted(
        set(dialogues).intersection(*teachers.values()), key=lambda s: int(s.split("_")[1])
    )
    runs: dict[str, list[Run]] = {
        "Lead-3": [lead_run(dialogues, eval_cfg["top_k"])],
        "Oracle": [oracle_run(labels / "oracle" / "test.jsonl")],
    }
    params: dict[str, float | None] = {"Lead-3": None, "Oracle": None}
    for t, run in teachers.items():
        runs[f"Teacher {t}"] = [run]
        params[f"Teacher {t}"] = parse_params(eval_cfg["teacher_params"][t])
    students = []
    for name in eval_cfg["students"]:
        seed_files = sorted((results / name).glob("seed_*/test_predictions.jsonl"))
        if not seed_files:
            log.warning("%s: no finished seeds, skipped", name)
            continue
        runs[name] = [labels_run(p) for p in seed_files]
        params[name] = json.loads((seed_files[0].parent / "metrics.json").read_text())["params"]
        students.append(name)
    reference = f"Teacher {eval_cfg['reference_teacher']}"

    eval_dir = results / "model1_eval"
    if all((eval_dir / f"{n.lower().replace(' ', '_')}.json").exists() for n in runs):
        log.info("metrics from %s", eval_dir)
        metrics = {
            n: json.loads((eval_dir / f"{n.lower().replace(' ', '_')}.json").read_text())["runs"]
            for n in runs
        }
    else:
        log.info("no complete %s: recomputing ROUGE and agreement (no BERTScore)", eval_dir)
        cfg = copy.deepcopy(eval_cfg)
        cfg["bertscore"]["enabled"] = False
        ref_run = runs[reference][0]
        metrics = {
            n: [evaluate_run(r, dialogues, ref_run, ids, cfg) for r in rs] for n, rs in runs.items()
        }
        ref_quality = np.mean([metrics[reference][0][t] for t in ROUGE_TYPES])
        for ms in metrics.values():
            for m in ms:
                m["retention"] = float(np.mean([m[t] for t in ROUGE_TYPES]) / ref_quality)
    return Model1Results(
        dialogues, ids, runs, metrics, params, reference, students, eval_cfg["top_k"]
    )


def per_dialogue_spearman(run: Run, reference: Run, ids: Sequence[str]) -> list[float]:
    """Spearman per dialogue against the reference (common prefix; undefined ones dropped)."""
    out = []
    for i in ids:
        p, t = run[i]["scores"], reference[i]["scores"]
        n = min(len(p), len(t))
        rho = spearman(p[:n], t[:n])
        if rho is not None:
            out.append(rho)
    return out


def pooled_spearman(res: Model1Results, name: str) -> list[float]:
    """Per-dialogue Spearman vs the reference teacher, pooled over the system's runs."""
    ref = res.runs[res.reference][0]
    return [rho for r in res.runs[name] for rho in per_dialogue_spearman(r, ref, res.ids)]


def selection_positions(res: Model1Results, name: str, bins: int) -> np.ndarray:
    """Share of selected utterances per relative-position bin in the dialogue (sums to 1)."""
    counts = np.zeros(bins)
    for run in res.runs[name]:
        for i in res.ids:
            n = len(res.dialogues[i]["utterances"])
            for j in selection(run[i], res.top_k):
                counts[min(int(j / n * bins), bins - 1)] += 1
    return counts / counts.sum()


def tag_share(res: Model1Results, tag: str) -> float:
    """Share of the reference teacher's utterance tags equal to ``tag``."""
    tags = [t for i in res.ids for t in res.runs[res.reference][0][i]["tags"]]
    return tags.count(tag) / len(tags)
