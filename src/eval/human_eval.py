"""Human review of both models on hand-written realistic inputs (Markdown report to read).

Usage::

    uv run python -m src.eval.human_eval --config configs/human_eval.yaml

Reads the inputs file (see its header) and writes a report with, for Model 2, input / model
output / reference per item, and for Model 1, per dialogue the scores of each student, its
three-line notes, and the same dialogue through the pipeline (utterances lowercased without
punctuation -> Model 2 -> Student A). Each model uses its seed with the best validation score.

The counts in the report (words with a different casing or punctuation than the reference,
expected utterances found in the top-k) are reading aids, not metrics: the inputs are few
and written by hand.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
from pathlib import Path
from typing import Any

import keras
import yaml

from src.config import load_config
from src.model1_scorer.infer import ScoredDialogue, dialogue_notes, score_dialogue
from src.model2_cleanup.infer import clean_text, input_words
from src.model2_cleanup.labels import label_text
from src.tokenizer.tokenizer import load_tokenizer
from src.training import utc_now

log = logging.getLogger(__name__)


def best_seed(results_dir: Path, name: str, metric: tuple[str, ...]) -> tuple[int, float]:
    """``(seed, value)`` of the seed with the highest validation ``metric``."""
    best: tuple[int, float] | None = None
    for path in sorted((results_dir / name).glob("seed_*/metrics.json")):
        value: Any = json.loads(path.read_text())
        for key in metric:
            value = value[key]
        seed = int(path.parent.name.split("_")[1])
        if best is None or float(value) > best[1]:
            best = (seed, float(value))
    if best is None:
        raise FileNotFoundError(f"No seed_*/metrics.json under {results_dir / name}")
    return best


def label_differences(output: str, reference: str) -> tuple[int, int]:
    """``(differing, compared)``: words aligned by form whose casing or punctuation differ."""
    out, ref = label_text(output), label_text(reference)
    matcher = difflib.SequenceMatcher(a=[w.form for w in out], b=[w.form for w in ref])
    differing = compared = 0
    for block in matcher.get_matching_blocks():
        for j in range(block.size):
            a, b = out[block.a + j], ref[block.b + j]
            compared += 1
            differing += (a.casing, a.punct) != (b.casing, b.punct)
    return differing, compared


def model2_section(items: list[dict], clean: Any) -> list[str]:
    """Report lines for the Model 2 items."""
    lines = ["## Model 2: punctuation and casing", ""]
    rows, blocks = [], []
    total_diff = total = 0
    for item in items:
        model_input = item.get("input") or " ".join(input_words(item["reference"]))
        output = clean(model_input)
        diff, n = label_differences(output, item["reference"])
        total_diff, total = total_diff + diff, total + n
        rows.append(f"| {item['id']} | {item['category']} | {diff} / {n} |")
        blocks += [
            f"**{item['id']}** ({item['category']})",
            "",
            f"- Input: `{model_input}`",
            f"- Model: {output}",
            f"- Reference: {item['reference']}",
            "",
        ]
    lines += [
        f"{len(items)} items. Words whose casing or punctuation differ from the reference "
        f"(words aligned by spelling): **{total_diff} / {total}** in total.",
        "",
        "| Item | Category | Differing words |",
        "|---|---|---|",
        *rows,
        "",
        *blocks,
    ]
    return lines


def picks(scored: ScoredDialogue) -> list[int]:
    """Selected utterances as 1-based numbers."""
    return [i + 1 for i in scored.selected]


def model1_section(
    dialogues: list[dict], students: dict[str, Any], pipeline: str, clean: Any
) -> list[str]:
    """Report lines for the Model 1 dialogues (``students``: label -> scoring function)."""
    names = list(students)
    header = "| Dialogue | Expected | " + " | ".join(names) + f" | Pipeline ({pipeline}) |"
    summary = [header, "|---" * (len(names) + 3) + "|"]
    hits = {n: 0 for n in [*names, "pipeline"]}
    n_expected = 0
    blocks: list[str] = []
    for d in dialogues:
        speakers = [int(s) for s, _ in d["turns"]]
        utterances = [u for _, u in d["turns"]]
        expect = d["expect"]
        n_expected += len(expect)
        scored = {n: score(speakers, utterances) for n, score in students.items()}
        cleaned = [clean(" ".join(input_words(u))) for u in utterances]
        scored_pipe = students[pipeline](speakers, cleaned)

        cells = []
        for n in names:
            found = len(set(picks(scored[n])) & set(expect))
            hits[n] += found
            cells.append(f"{picks(scored[n])} ({found}/{len(expect)})")
        found = len(set(picks(scored_pipe)) & set(expect))
        hits["pipeline"] += found
        summary.append(
            f"| {d['id']} | {expect} | " + " | ".join(cells) + f" | {picks(scored_pipe)} "
            f"({found}/{len(expect)}) |"
        )

        table = [
            "| # | Speaker | Utterance | Expected | "
            + " | ".join(names)
            + " | Pipeline: Model 2 output | Pipeline score |",
            "|---" * (len(names) + 6) + "|",
        ]
        for i, (s, u) in enumerate(zip(speakers, utterances, strict=True)):
            cols = []
            for n in names:
                sd = scored[n]
                mark = "**" if i in sd.selected else ""
                tag = f" {sd.tags[i]}" if sd.tags else ""
                cols.append(f"{mark}{sd.scores[i]:.2f}{mark}{tag}")
            pmark = "**" if i in scored_pipe.selected else ""
            table.append(
                f"| {i + 1} | {s} | {u} | {'✓' if i + 1 in expect else ''} | "
                + " | ".join(cols)
                + f" | {cleaned[i]} | {pmark}{scored_pipe.scores[i]:.2f}{pmark} |"
            )
        blocks += [
            f"### {d['id']} ({d['category']})",
            "",
            f"Expected notes: {d['note']}",
            "",
            *table,
            "",
            "Scores in bold are the top-k. Tags are the student's predicted tags.",
            "",
            *[
                line
                for n in names
                for line in (
                    f"**Notes, {n}:**",
                    "",
                    dialogue_notes(scored[n], speakers, utterances),
                    "",
                )
            ],
            f"**Notes, pipeline (Model 2 -> {pipeline}):**",
            "",
            dialogue_notes(scored_pipe, speakers, cleaned),
            "",
        ]
    totals = ", ".join(f"{n} {hits[n]}/{n_expected}" for n in [*names, "pipeline"])
    return [
        "## Model 1: dialogue notes",
        "",
        f"{len(dialogues)} dialogues. Expected utterances found in the top-k: {totals}.",
        "",
        *summary,
        "",
        *blocks,
    ]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/human_eval.yaml"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    keras.utils.set_random_seed(cfg["seed"])
    inputs = yaml.safe_load(Path(cfg["inputs"]).read_text())
    tokenizer = load_tokenizer(cfg["tokenizer_dir"])
    used = []

    m2_cfg = load_config(cfg["model2_config"])
    seed, value = best_seed(
        Path(m2_cfg["paths"]["results_dir"]), m2_cfg["name"], ("validation", "punct", "macro_f1")
    )
    ckpt = Path(m2_cfg["paths"]["output_dir"]) / f"seed_{seed}" / "best.keras"
    model2 = keras.saving.load_model(ckpt, compile=False)
    used.append(f"| Model 2 | {m2_cfg['name']} | {seed} | punct macro-F1 {value:.3f} |")

    def clean(text: str) -> str:
        return clean_text(
            model2, tokenizer, text, m2_cfg["data"]["window_words"], m2_cfg["data"]["max_len"]
        )

    students = {}
    for path in cfg["model1_configs"]:
        s_cfg = load_config(path)
        seed, value = best_seed(
            Path(s_cfg["paths"]["results_dir"]), s_cfg["name"], ("validation", "spearman")
        )
        ckpt = Path(s_cfg["paths"]["output_dir"]) / f"seed_{seed}" / "best.keras"
        model = keras.saving.load_model(ckpt, compile=False)
        used.append(f"| Model 1 | {s_cfg['name']} | {seed} | Spearman {value:.3f} |")

        def score(speakers, utterances, model=model, s_cfg=s_cfg) -> ScoredDialogue:
            return score_dialogue(
                model,
                tokenizer,
                speakers,
                utterances,
                s_cfg["model"]["max_len"],
                s_cfg["inference"]["top_k"],
            )

        students[s_cfg["name"]] = score

    lines = [
        "# Human review: realistic inputs",
        "",
        f"Generated {utc_now()} by `python -m src.eval.human_eval` from `{cfg['inputs']}` "
        "(hand-written, not from any dataset). Counts below are reading aids, not metrics.",
        "",
        "| Model | Run | Seed | Validation |",
        "|---|---|---|---|",
        *used,
        "",
        *model2_section(inputs["model2"], clean),
        *model1_section(inputs["model1"], students, next(iter(students)), clean),
    ]
    report = Path(cfg["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", report)


if __name__ == "__main__":
    main()
