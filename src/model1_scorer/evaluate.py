"""Stage 7c: evaluate every Model 1 system on DialogSum test (PLAN.md section 5.6).

Usage (local CPU, after the Kaggle runs are fetched)::

    uv run --extra eval python -m src.model1_scorer.evaluate --config configs/model1_eval.yaml

Systems: lead-3 and the oracle (reference rows), both teachers (their own top-3), and every
student seed (``test_predictions.jsonl`` from :mod:`train`). All use the same inference rule
(:mod:`notes`) and the same test dialogues: those labelled by every teacher.

Metrics per system:
    ROUGE-1/2/L F1 (mean over the 3 references) and BERTScore F1 of the notes text;
    retention = mean ROUGE(1, 2, L) of the system / that of the reference (7B) teacher;
    agreement with the reference teacher: per-dialogue Spearman of the scores, top-k Jaccard;
    ``decision`` / ``action_item`` F1 against the reference teacher's tags (agreement with the
    teacher, not ground truth);
    efficiency: parameters, ``.keras`` size and CPU latency per dialogue (batch 1) for the
    students; the teachers' labelling throughput (GPU, batched) from their ``stats.json``.
Writes ``<results_dir>/model1_eval/<system>.json`` and ``<results_dir>/tables/model1.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.data.io import read_jsonl
from src.eval.metrics import classification_report, jaccard, mean_spearman
from src.eval.summary import ROUGE_TYPES, bertscore, rouge
from src.model1_scorer.data import TAGS
from src.model1_scorer.notes import notes_text, select_top_k

log = logging.getLogger(__name__)
Run = dict[str, dict[str, Any]]  # dialogue id -> {"scores", "tags", "selected"}
TAG_CLASSES = ("decision", "action_item")


def lead_run(dialogues: dict[str, dict], k: int) -> Run:
    """First k utterances (scores decrease with position)."""
    run = {}
    for i, d in dialogues.items():
        n = len(d["utterances"])
        run[i] = {"scores": [float(n - j) for j in range(n)], "selected": list(range(min(k, n)))}
    return run


def oracle_run(path: Path) -> Run:
    """The oracle's own selection (its 0/1 labels as scores)."""
    return {r["id"]: {"scores": r["labels"], "selected": r["selected"]} for r in read_jsonl(path)}


def labels_run(path: Path) -> Run:
    """Teacher labels or student predictions: per-utterance scores (and tags if present)."""
    return {r["id"]: {"scores": r["scores"], "tags": r.get("tags")} for r in read_jsonl(path)}


def selection(entry: dict[str, Any], k: int) -> list[int]:
    return entry.get("selected") or select_top_k(entry["scores"], k)


def evaluate_run(
    run: Run, dialogues: dict[str, dict], reference: Run, ids: list[str], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Summary quality and agreement with the reference teacher on ``ids``."""
    k = cfg["top_k"]
    texts, refs = [], []
    for i in ids:
        tags = run[i].get("tags")
        texts.append(notes_text(dialogues[i]["utterances"], selection(run[i], k), tags))
        refs.append(dialogues[i]["summaries"])
    metrics: dict[str, Any] = {"dialogues": len(ids), **rouge(texts, refs, cfg["use_stemmer"])}
    if cfg["bertscore"]["enabled"]:
        metrics["bertscore"] = bertscore(
            texts, refs, cfg["bertscore"]["model_type"], cfg["bertscore"]["batch_size"]
        )
    # Agreement: compare on the utterances both sides scored (students may have truncated).
    pairs = [(run[i]["scores"], reference[i]["scores"]) for i in ids]
    pairs = [(p[: min(len(p), len(t))], t[: min(len(p), len(t))]) for p, t in pairs]
    metrics["spearman"] = mean_spearman([p for p, _ in pairs], [t for _, t in pairs])
    metrics["jaccard"] = float(
        np.mean([jaccard(selection(run[i], k), selection(reference[i], k)) for i in ids])
    )
    if all(run[i].get("tags") for i in ids):
        pred, true = [], []
        for i in ids:
            n = min(len(run[i]["tags"]), len(reference[i]["tags"]))
            pred += [TAGS.index(t) for t in run[i]["tags"][:n]]
            true += [TAGS.index(t) for t in reference[i]["tags"][:n]]
        report = classification_report(np.array(true), np.array(pred), TAGS, TAG_CLASSES)
        metrics["tags"] = {t: report["per_class"][t] for t in TAG_CLASSES}
    return metrics


def cpu_latency_ms(checkpoint: Path, processed_dir: Path, n: int) -> float:
    """Median wall time of one dialogue (batch 1) on this machine's CPU."""
    import keras

    from src.model1_scorer.data import StudentSet
    from src.model1_scorer.train import model_inputs

    model = keras.saving.load_model(checkpoint)
    data = StudentSet.load(processed_dir / "test")
    rows = np.arange(min(n, len(data)) + 1)
    times = []
    for j, row in enumerate(rows):
        inputs = model_inputs(data, np.array([row]))
        start = time.perf_counter()
        model.predict_on_batch(inputs)
        if j:  # the first call compiles
            times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


def teacher_throughput_ms(stats_path: Path) -> float | None:
    """Labelling time per dialogue on Kaggle (GPU, batched), from the teacher's stats.json."""
    if not stats_path.exists():
        return None
    splits = json.loads(stats_path.read_text())["splits"].values()
    seconds = sum(s["seconds_this_session"] for s in splits)
    done = sum(s["this_session"]["labelled"] + s["this_session"]["failed"] for s in splits)
    return seconds / done * 1000 if done else None


def mean_std(values: list[float], digits: int = 3) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


COLUMNS = [
    ("ROUGE-1", lambda m: m["rouge1"]),
    ("ROUGE-2", lambda m: m["rouge2"]),
    ("ROUGE-L", lambda m: m["rougeL"]),
    ("BERTScore", lambda m: m.get("bertscore")),
    ("Retention", lambda m: m.get("retention")),
    ("Spearman vs ref", lambda m: m.get("spearman")),
    ("Top-k Jaccard vs ref", lambda m: m.get("jaccard")),
    ("Decision F1", lambda m: m["tags"]["decision"]["f1"] if "tags" in m else None),
    ("Action F1", lambda m: m["tags"]["action_item"]["f1"] if "tags" in m else None),
]


def results_table(rows: list[dict[str, Any]], cfg: dict[str, Any], n: int) -> str:
    """Markdown table: one row per system, mean ± std over seeds for students."""
    header = ["System", "Params", *[c for c, _ in COLUMNS], "Latency (ms)", "Size (MB)"]
    lines = [
        "# Model 1: extractive notes on DialogSum test",
        "",
        f"{n} test dialogues (labelled by every teacher). Top-{cfg['top_k']} utterances per "
        f"dialogue. Reference teacher: `{cfg['reference_teacher']}` (agreement and tag F1 are "
        "agreement with this teacher, not ground truth). Students: mean ± std over seeds. "
        "Latency: students on this machine's CPU (batch 1); teachers on Kaggle GPU (batched "
        "vLLM labelling).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells = [mean_std([f(m) for m in row["runs"]]) for _, f in COLUMNS]
        cells = [row["name"], row["params"], *cells, row.get("latency", "–"), row.get("size", "–")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Evaluate all systems and write the per-system JSON files and the results table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/model1_eval.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    labels, results = Path(paths["labels_dir"]), Path(paths["results_dir"])
    dialogues = {
        r["id"]: r for r in read_jsonl(Path(paths["raw_dir"]) / "dialogsum" / "test.jsonl")
    }

    teachers = {t: labels_run(labels / t / "test.jsonl") for t in cfg["teachers"]}
    reference = teachers[cfg["reference_teacher"]]
    students: dict[str, dict[int, Run]] = {}
    for name in cfg["students"]:
        students[name] = {
            int(p.parent.name.split("_")[1]): labels_run(p)
            for p in sorted((results / name).glob("seed_*/test_predictions.jsonl"))
        }
    ids = sorted(
        set(dialogues).intersection(*teachers.values()), key=lambda s: int(s.split("_")[1])
    )
    for name, seeds in students.items():
        for seed, run in seeds.items():
            if missing := set(ids) - set(run):
                raise ValueError(f"{name} seed {seed}: no predictions for {sorted(missing)[:3]}")
    log.info("evaluating on %d test dialogues", len(ids))

    def run_metrics(run: Run) -> dict[str, Any]:
        return evaluate_run(run, dialogues, reference, ids, cfg)

    systems: list[dict[str, Any]] = [
        {"name": "Lead-3", "params": "–", "runs": [run_metrics(lead_run(dialogues, cfg["top_k"]))]},
        {
            "name": "Oracle",
            "params": "–",
            "runs": [run_metrics(oracle_run(labels / "oracle" / "test.jsonl"))],
        },
    ]
    for t, run in teachers.items():
        ms = teacher_throughput_ms(labels / t / "stats.json")
        systems.append(
            {
                "name": f"Teacher {t}",
                "params": cfg["teacher_params"][t],
                "runs": [run_metrics(run)],
                "latency": f"{ms:.0f} (GPU)" if ms else "–",
            }
        )
    for name, seeds in students.items():
        if not seeds:
            log.warning("%s: no finished seeds, skipped", name)
            continue
        runs = [run_metrics(seeds[s]) for s in sorted(seeds)]
        first = results / name / f"seed_{min(seeds)}"
        params = json.loads((first / "metrics.json").read_text())["params"]
        row = {"name": name, "params": f"{params / 1e6:.1f}M", "runs": runs}
        ckpt = Path(cfg["students"][name]) / f"seed_{min(seeds)}" / "best.keras"
        if ckpt.exists():
            row["size"] = f"{ckpt.stat().st_size / 1e6:.1f}"
            ms = cpu_latency_ms(ckpt, Path(paths["processed_dir"]), cfg["latency_dialogues"])
            row["latency"] = f"{ms:.1f}"
        systems.append(row)

    ref_row = next(s for s in systems if s["name"] == f"Teacher {cfg['reference_teacher']}")
    ref_quality = np.mean([ref_row["runs"][0][t] for t in ROUGE_TYPES])
    out = results / "model1_eval"
    out.mkdir(parents=True, exist_ok=True)
    for system in systems:
        for m in system["runs"]:
            m["retention"] = float(np.mean([m[t] for t in ROUGE_TYPES]) / ref_quality)
        slug = system["name"].lower().replace(" ", "_")
        (out / f"{slug}.json").write_text(json.dumps(system, indent=2) + "\n")
    table = results / "tables" / "model1.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(results_table(systems, cfg, len(ids)))
    print(table.read_text())


if __name__ == "__main__":
    main()
