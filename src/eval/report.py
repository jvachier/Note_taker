"""Interactive results report for both models: figures, what they show, and a verdict each.

Usage::

    uv run python -m src.eval.report --config configs/report.yaml

Builds every figure of :mod:`src.eval.figures` from the saved results (:mod:`results_data`),
explains what each one shows, and attaches a verdict (good / mixed / poor, or context) that is
computed from the numbers with the thresholds in the config, so it stays true when results
change. Writes one self-contained HTML page (Plotly embedded once, works offline).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.config import load_config
from src.eval import figures as F
from src.eval.aggregate import load_seeds
from src.eval.results_data import (
    DATASET_NAMES,
    MODEL2_CLASSES,
    Model1Results,
    display_name,
    epoch_stats,
    load_model1,
    model2_test_sets,
    model2_value,
    per_dialogue_spearman,
    pooled_spearman,
    read_history,
    seed_histories,
    selection_positions,
    tag_share,
)
from src.model1_scorer.evaluate import selection

log = logging.getLogger(__name__)

LEVELS = {  # icon + label always accompany the colour
    "good": ("✓", "Good"),
    "mixed": ("!", "Mixed"),
    "poor": ("✕", "Poor"),
    "info": ("i", "Context"),
}


@dataclass
class Verdict:
    """Outcome of one figure: a level, a one-sentence result and supporting facts."""

    level: str
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class Section:
    """One figure of the report with its explanation and verdict."""

    key: str
    model: str
    headline: str
    what: str
    verdict: Verdict
    figure: go.Figure


def pct(x: float) -> str:
    return f"{x:.0%}"


# ----------------------------------------------------------------------------- Model 2


def model2_noise_verdict(seeds: dict[int, dict[str, Any]], th: dict[str, Any]) -> Verdict:
    """Does the model beat the baseline at every noise level, and how much does noise cost?"""
    beats, drops, details = True, [], []
    for dataset, entries in model2_test_sets(seeds).items():
        (n0, k0), (n_max, k_max) = entries[0], entries[-1]
        for _, key in entries:
            for path in (("punct", "macro_f1"), ("casing", "accuracy")):
                model = model2_value(seeds, key, "model", path)[0]
                beats &= model > model2_value(seeds, key, "baseline", path)[0]
        p0 = model2_value(seeds, k0, "model", ("punct", "macro_f1"))[0]
        p_max = model2_value(seeds, k_max, "model", ("punct", "macro_f1"))[0]
        c0 = model2_value(seeds, k0, "model", ("casing", "accuracy"))[0]
        c_max = model2_value(seeds, k_max, "model", ("casing", "accuracy"))[0]
        base = model2_value(seeds, k0, "baseline", ("punct", "macro_f1"))[0]
        drops.append(1 - p_max / p0)
        details.append(
            f"{DATASET_NAMES.get(dataset, dataset)}: punctuation macro-F1 {p0:.3f} → {p_max:.3f} "
            f"from {pct(n0)} to {pct(n_max)} noise (rule baseline {base:.3f}); casing accuracy "
            f"{c0:.3f} → {c_max:.3f}."
        )
    worst = max(drops)
    level = "poor" if not beats else "good" if worst <= th["noise_drop_ok"] else "mixed"
    summary = (
        f"Model 2 {'beats' if beats else 'does not always beat'} the rule baseline at every noise "
        f"level. Noisy input costs it at most {pct(worst)} of its punctuation F1, and casing "
        "barely moves."
    )
    return Verdict(level, summary, details)


def model2_per_class_verdict(
    seeds: dict[int, dict[str, Any]], noise: float, th: dict[str, Any]
) -> Verdict:
    """Which classes are strong and which are weak?"""
    strong, moderate, weak, rare = [], [], [], []
    for dataset, entries in model2_test_sets(seeds).items():
        key = next(k for n, k in entries if np.isclose(n, noise))
        report = seeds[min(seeds)]["test"][key]["model"]
        for cls, path in MODEL2_CLASSES:
            f1 = model2_value(seeds, key, "model", path)[0]
            name = f"{cls} on {DATASET_NAMES.get(dataset, dataset).removesuffix(' test')}"
            support = report[path[0]]["per_class"][path[2]].get("support")
            if support is not None and support < th["min_support"]:
                rare.append(f"{name} (F1 {f1:.2f} on only {support} examples)")
            elif f1 >= th["good_f1"]:
                strong.append(name)
            elif f1 >= th["poor_f1"]:
                moderate.append(f"{name} ({f1:.2f})")
            else:
                weak.append(f"{name} ({f1:.2f})")
    total = len(strong) + len(moderate) + len(weak)
    if not weak and not moderate:
        level = "good"
    elif len(weak) > total / 2:
        level = "poor"
    else:
        level = "mixed"
    summary = f"{len(strong)} of {total} class / test-set pairs reach F1 ≥ {th['good_f1']:.2f}."
    if weak:
        summary += f" Weak: {', '.join(weak)}."
    details = []
    if rare:
        details.append(
            f"Too few examples to judge (< {th['min_support']}): {', '.join(rare)}; not counted."
        )
    if moderate:
        details.append(
            f"Moderate (F1 {th['poor_f1']:.2f}–{th['good_f1']:.2f}): {', '.join(moderate)}."
        )
    details.append(
        "The rule baseline (capitalise sentence starts, period at the end) scores 0 on commas, "
        "question marks and UPPER by construction."
    )
    return Verdict(level, summary, details)


def converged(stats: np.ndarray, th: dict[str, Any]) -> tuple[bool, float]:
    """Gain of the mean over the last ``plateau_epochs`` epochs and whether it is a plateau."""
    k = min(th["plateau_epochs"], len(stats) - 1)
    gain = float(stats[-1, 0] - stats[-1 - k, 0])
    return gain < th["plateau_gain"], gain


def model2_training_verdict(histories: list[dict[str, np.ndarray]], th: dict) -> Verdict:
    """Has training converged, and do loss and F1 agree?"""
    f1 = epoch_stats(histories, "val_punct_macro_f1")
    loss = epoch_stats(histories, "val_loss")
    flat, gain = converged(f1, th)
    best = int(np.argmax(f1[:, 0])) + 1
    details = [
        f"Best validation punctuation macro-F1 {f1[:, 0].max():.3f} at epoch {best} of {len(f1)}; "
        f"gain over the last {th['plateau_epochs']} epochs: {gain:+.4f}."
    ]
    low = int(np.argmin(loss[:, 0])) + 1
    if low < len(loss) - 1:
        details.append(
            f"Validation loss is lowest at epoch {low} and rises after, while F1 keeps improving: "
            "the model grows more confident on its mistakes. Checkpoints are selected on "
            "macro-F1, so the kept model is the best one on the metric we report."
        )
    summary = (
        f"Training has converged (validation macro-F1 {f1[-1, 0]:.3f}, seeds agree within "
        f"±{f1[-1, 1]:.3f})."
        if flat
        else f"Validation macro-F1 is still rising (+{gain:.3f} over the last epochs): more "
        "epochs would likely help."
    )
    return Verdict("good" if flat else "mixed", summary, details)


# ----------------------------------------------------------------------------- Model 1


def mlm_verdict(history: dict[str, np.ndarray], held_out: dict[str, Any], th: dict) -> Verdict:
    """Did pretraining learn without overfitting?"""
    val = history["val_loss"]
    overfit = val[-1] > val.min() * 1.01
    acc = np.stack([history["val_acc"], np.zeros_like(history["val_acc"])], axis=1)
    flat, gain = converged(acc, th)
    accs = ", ".join(
        f"{v['accuracy']:.2f} on {k.split('_')[0].replace('wikitext', 'WikiText').replace('dialogsum', 'DialogSum')}"  # noqa: E501
        for k, v in held_out.items()
    )
    summary = f"Held-out masked-token accuracy: {accs}. " + (
        "Validation loss went back up: the encoder started to overfit."
        if overfit
        else "Validation loss is still falling at the last epoch, so there is no overfitting."
    )
    details = [
        "Masked-token accuracy = share of hidden (masked) tokens the encoder predicts exactly; "
        "it is the pretraining signal the students start from, not a downstream metric."
    ]
    if not flat:
        details.append(
            f"Accuracy still rose by {gain:.3f} over the last {th['plateau_epochs']} epochs: "
            "longer pretraining would likely help the students a little."
        )
    return Verdict("mixed" if overfit else "good", summary, details)


def student_training_verdict(
    histories: dict[str, list[dict[str, np.ndarray]]], th: dict
) -> Verdict:
    """Which students train smoothly and which are unstable?"""
    unstable, details = [], []
    for name, hist in histories.items():
        stats = epoch_stats(hist, "val_spearman")
        best, last = stats[:, 0].max(), stats[-1, 0]
        line = (
            f"{display_name(name)}: best {best:.3f} at epoch {int(np.argmax(stats[:, 0])) + 1}, "
            f"{len(stats)} epochs"
        )
        if best - last > th["unstable_drop"]:
            unstable.append(display_name(name))
            line += f"; falls to {last:.3f} by the end (early stopping keeps the best epoch)"
        details.append(line + ".")
    details.append(
        "Each student is scored against its own targets (C against 0/1 oracle labels), so the "
        "levels are not comparable across students; the shapes are."
    )
    if unstable:
        summary = (
            f"{', '.join(unstable)} {'is' if len(unstable) == 1 else 'are'} unstable: validation "
            "agreement peaks early and then degrades. The other students converge smoothly."
        )
        return Verdict("mixed", summary, details)
    return Verdict("good", "Every student converges smoothly.", details)


def _rouge_l(res: Model1Results, name: str) -> float:
    return res.value(name, lambda m: m["rougeL"])[0]


def quality_verdict(res: Model1Results, main: str, th: dict) -> Verdict:
    """Is the main student as good as its teacher, and better than Lead-3?"""
    ret = res.value(main, lambda m: m["retention"])[0]
    s, t, lead = _rouge_l(res, main), _rouge_l(res, res.reference), _rouge_l(res, "Lead-3")
    above_lead = [n for n in res.students if _rouge_l(res, n) > lead]
    level = "poor" if ret < th["retention_target"] else "good" if s >= lead else "mixed"
    summary = (
        f"{display_name(main)} matches its teacher (ROUGE-L {s:.3f} vs {t:.3f}, retention "
        f"{ret:.2f}), "
        + (
            f"but neither beats simply taking the first three utterances (Lead-3, {lead:.3f})."
            if s < lead and t < lead
            else f"and Lead-3 scores {lead:.3f}."
        )
    )
    details = []
    if t < lead:
        details.append(
            f"The ceiling comes from the teacher: {display_name(res.reference)}'s own top-3 scores "
            "below Lead-3, and a distilled student can only be as good as what it imitates."
        )
    for n in above_lead:
        details.append(
            f"{display_name(n)} beats Lead-3 (ROUGE-L {_rouge_l(res, n):.3f}): it is trained "
            "directly on oracle labels, i.e. on what ROUGE rewards."
        )
    details.append(
        f"The oracle ({_rouge_l(res, 'Oracle'):.3f}) is the best any 3-utterance selection can do."
    )
    if not all("bertscore" in m for ms in res.metrics.values() for m in ms):
        details.append(
            "BERTScore is not shown yet: run `uv run --extra eval python -m "
            "src.model1_scorer.evaluate` and rebuild this report."
        )
    return Verdict(level, summary, details)


def agreement_verdict(res: Model1Results, main: str, ablation: dict, th: dict) -> Verdict:
    """Does the main student reproduce the teacher, and does MLM pretraining help?"""

    def v(name: str, key: str) -> float:
        return res.value(name, lambda m: m[key])[0]

    ret, rho = v(main, "retention"), v(main, "spearman")
    ok_ret, ok_rho = ret >= th["retention_target"], rho >= th["good_spearman"]
    level = "good" if ok_ret and ok_rho else ("mixed" if ok_ret or ok_rho else "poor")
    summary = (
        f"{display_name(main)} keeps {pct(ret)} of the teacher's ROUGE (target "
        f"{pct(th['retention_target'])}) and ranks utterances like it (Spearman {rho:.2f})."
    )
    other_teachers = [n for n in res.runs if n.startswith("Teacher ") and n != res.reference]
    for t in other_teachers:
        summary += f" For comparison, {display_name(t)} reaches {v(t, 'spearman'):.2f}."
    a, b = ablation["with_mlm"], ablation["without_mlm"]
    details = []
    if a in res.runs and b in res.runs:
        details.append(
            f"MLM pretraining helps: without it ({display_name(b)}) Spearman drops from "
            f"{v(a, 'spearman'):.2f} to {v(b, 'spearman'):.2f} and top-3 overlap from "
            f"{v(a, 'jaccard'):.2f} to {v(b, 'jaccard'):.2f}, at equal ROUGE."
        )
    for n in res.students:
        if n not in (main, b):
            details.append(
                f"{display_name(n)}: Spearman {v(n, 'spearman'):.2f}, top-3 overlap "
                f"{v(n, 'jaccard'):.2f}, retention {v(n, 'retention'):.2f}."
            )
    details.append(
        "Retention above 1 (Lead-3, Oracle, oracle-trained student) means a higher ROUGE than "
        "the teacher, not agreement with it. Agreement is with the teacher, not ground truth."
    )
    return Verdict(level, summary, details)


def spearman_distribution_verdict(res: Model1Results, main: str, th: dict) -> Verdict:
    """How consistent is the main student's agreement across dialogues?"""
    rhos = np.array(pooled_spearman(res, main))
    med = float(np.median(rhos))
    q1, q3 = np.percentile(rhos, [25, 75])
    level = (
        "good"
        if med >= th["good_spearman"]
        else ("mixed" if med >= th["poor_spearman"] else "poor")
    )
    summary = (
        f"On a typical dialogue {display_name(main)} agrees well with the teacher (median "
        f"Spearman {med:.2f}, half of dialogues between {q1:.2f} and {q3:.2f}); it disagrees "
        f"(Spearman < 0) on {pct(float(np.mean(rhos < 0)))} of dialogues."
    )
    details = [
        f"{display_name(n)}: median {np.median(pooled_spearman(res, n)):.2f}."
        for n in res.runs
        if n not in (main, res.reference, "Lead-3", "Oracle")
    ]
    return Verdict(level, summary, details)


def tags_verdict(res: Model1Results, main: str, th: dict) -> Verdict:
    """Does the main student reproduce the teacher's decision / action-item tags?"""
    f1 = {
        t: res.value(main, lambda m, t=t: m["tags"][t]["f1"])[0]
        for t in ("decision", "action_item")
    }
    if min(f1.values()) >= th["good_f1"]:
        level = "good"
    elif max(f1.values()) >= th["poor_f1"]:
        level = "mixed"
    else:
        level = "poor"
    summary = (
        f"Tag agreement is {'weak' if level == 'poor' else 'partial'}: {display_name(main)} "
        f"reaches decision F1 {f1['decision']:.2f} and action-item F1 {f1['action_item']:.2f} "
        "against the teacher's tags."
    )
    details = [
        f"Decisions are only {tag_share(res, 'decision'):.1%} of the teacher's tags and action "
        f"items {tag_share(res, 'action_item'):.1%}: rare classes with few training examples.",
    ]
    for t in (n for n in res.runs if n.startswith("Teacher ") and n != res.reference):
        details.append(
            f"The tags themselves are unreliable: {display_name(t)} agrees with "
            f"{display_name(res.reference)} on decisions at F1 "
            f"{res.value(t, lambda m: m['tags']['decision']['f1'])[0]:.2f}."
        )
    return Verdict(level, summary, details)


def positions_verdict(res: Model1Results, systems: list[str], bins: int) -> Verdict:
    """Where each system picks from (context for the ROUGE results)."""
    first = {s: selection_positions(res, s, bins)[0] for s in systems if s in res.runs}
    parts = [f"{display_name(s)} {pct(v)}" for s, v in first.items()]
    summary = f"Share of picks in the first {pct(1 / bins)} of the dialogue: {', '.join(parts)}."
    details = []
    if "Oracle" in first and res.reference in first:
        details.append(
            f"The oracle, which maximises ROUGE, leans to the opening ({pct(first['Oracle'])}); "
            f"the teacher spreads its picks ({pct(first[res.reference])}). DialogSum summaries "
            "draw on how a conversation starts, which is why Lead-3 is a strong baseline and "
            "why the teacher-distilled students score lower on ROUGE."
        )
    return Verdict("info", summary, details)


def example_verdict(res: Model1Results, main: str, dialogue_id: str) -> Verdict:
    """How the main student compares with the teacher on the example dialogue."""
    student, teacher = res.runs[main][0][dialogue_id], res.runs[res.reference][0][dialogue_id]
    picks, ref_picks = set(selection(student, res.top_k)), set(selection(teacher, res.top_k))
    rho = per_dialogue_spearman(res.runs[main][0], res.runs[res.reference][0], [dialogue_id])
    corr = f" Spearman with the teacher's scores on this dialogue: {rho[0]:.2f}." if rho else ""
    return Verdict(
        "info",
        f"On this dialogue {display_name(main)} picks {len(picks & ref_picks)} of the teacher's "
        f"{len(ref_picks)} utterances.{corr}",
        [
            "Hover a cell to read the utterance. Teacher scores (0–10) are shown divided by 10; "
            "oracle labels are 0/1; students are seed 0."
        ],
    )


def size_verdict(res: Model1Results, main: str, th: dict) -> Verdict:
    """What does the student cost relative to the teacher?"""
    ratio = res.params[res.reference] / res.params[main]
    ret = res.value(main, lambda m: m["retention"])[0]
    level = "good" if ret >= th["retention_target"] else "poor"
    return Verdict(
        level,
        f"{display_name(main)} is {ratio:,.0f}× smaller than {display_name(res.reference)} "
        f"({res.params[main] / 1e6:.1f}M parameters) and keeps {pct(ret)} of its ROUGE.",
        ["It runs on a laptop CPU; the teacher needs a GPU."],
    )


# ----------------------------------------------------------------------------- build


def build_sections(cfg: dict[str, Any]) -> list[Section]:
    """Every section whose inputs exist."""
    th, m2, m1 = cfg["verdicts"], cfg["model2"], cfg["model1"]
    eval_cfg = load_config(Path(m1["eval_config"]))
    results = Path(eval_cfg["paths"]["results_dir"])
    sections: list[Section] = []

    def add(key, model, headline, what, verdict: Callable[[], Verdict], fig: Callable):
        sections.append(Section(key, model, headline, what, verdict(), fig()))

    if (results / m2["exp"]).exists():
        seeds = load_seeds(results / m2["exp"])
        hist = seed_histories(results / m2["exp"])
        add(
            "m2-noise",
            "Model 2",
            "Robustness to noisy input",
            "Each line is a test score as more words of the input are corrupted (substituted, "
            "deleted or inserted). Top: punctuation (macro-F1 over none / comma / period / "
            "question). Bottom: casing accuracy. Blue is our model, grey the rule baseline. "
            "Higher is better; a flat blue line means robust.",
            lambda: model2_noise_verdict(seeds, th),
            lambda: F.model2_noise(seeds),
        )
        add(
            "m2-classes",
            "Model 2",
            "Strengths and weaknesses per class",
            "F1 per punctuation mark and per casing class on clean input. F1 balances missed "
            "marks against wrongly added ones; 1.0 is perfect.",
            lambda: model2_per_class_verdict(seeds, m2["per_class_noise"], th),
            lambda: F.model2_per_class(seeds, m2["per_class_noise"]),
        )
        add(
            "m2-training",
            "Model 2",
            "Training behaviour",
            "Loss and validation scores per epoch, averaged over seeds (the wash is ± one std). "
            "Converged training flattens out; a widening gap between train and validation loss "
            "means overfitting.",
            lambda: model2_training_verdict(hist, th),
            lambda: F.model2_training(hist),
        )

    mlm_hist = Path(m1["mlm_dir"]) / "history.csv"
    if mlm_hist.exists():
        h = read_history(mlm_hist)
        held = json.loads((Path(m1["mlm_dir"]) / "metrics.json").read_text())["held_out"]
        add(
            "m1-mlm",
            "Model 1",
            "Encoder pretraining (MLM)",
            "Before learning to score utterances, the encoder learns language by predicting "
            "hidden words (masked language modelling) on WikiText-103 and DialogSum.",
            lambda: mlm_verdict(h, held, th),
            lambda: F.mlm_curves(h),
        )

    if not (Path(eval_cfg["paths"]["labels_dir"]) / eval_cfg["reference_teacher"]).exists():
        log.warning("no teacher labels: Model 1 system sections skipped")
        return sections
    res = load_model1(eval_cfg)
    main = m1["main_student"]
    sizes = eval_cfg["teacher_params"]
    strong = sizes[eval_cfg["reference_teacher"]]
    weak = next((v for t, v in sizes.items() if t != eval_cfg["reference_teacher"]), "small")
    hists = {n: seed_histories(results / n) for n in res.students}
    add(
        "m1-training",
        "Model 1",
        "Student training",
        "Validation agreement (Spearman) of each student with its own targets, per epoch, "
        f"averaged over seeds. A: {strong} teacher; B: {weak} teacher; C: oracle labels; D: like "
        "A but without MLM pretraining; E: oracle importance with teacher tags.",
        lambda: student_training_verdict(hists, th),
        lambda: F.student_training(hists),
    )
    add(
        "m1-quality",
        "Model 1",
        "Note quality against human summaries",
        f"Each system picks its top-{res.top_k} utterances per dialogue and is scored against "
        "the DialogSum human summaries with ROUGE (word overlap). Grey: reference rows (Lead-3 "
        "= first three utterances; Oracle = best possible pick). Orange: Qwen teachers. Blue: "
        "our 17M-parameter students.",
        lambda: quality_verdict(res, main, th),
        lambda: F.model1_quality(res),
    )
    add(
        "m1-agreement",
        "Model 1",
        f"Agreement with the {strong} teacher",
        f"How closely each system reproduces the {strong} teacher. Retention = its ROUGE "
        "divided by the teacher's. Spearman = how similarly it ranks the utterances of a "
        "dialogue (1 = same order, 0 = unrelated). Top-3 overlap = shared picks (Jaccard).",
        lambda: agreement_verdict(res, main, m1["ablation"], th),
        lambda: F.model1_agreement(res, th["retention_target"]),
    )
    add(
        "m1-spread",
        "Model 1",
        "Agreement dialogue by dialogue",
        "The same Spearman correlation, now one value per test dialogue. The box spans the "
        "middle half of dialogues, the line inside is the median, whiskers show the range.",
        lambda: spearman_distribution_verdict(res, main, th),
        lambda: F.model1_spearman_distribution(res),
    )
    add(
        "m1-tags",
        "Model 1",
        "Decision and action-item tags",
        "Every utterance also gets a tag (none / key point / decision / action item). F1 here "
        f"measures agreement with the {strong} teacher's tags; there are no human tags.",
        lambda: tags_verdict(res, main, th),
        lambda: F.model1_tags(res),
    )
    add(
        "m1-positions",
        "Model 1",
        "Where the picked utterances come from",
        "Position of the picked utterances inside each dialogue, from the start (0%) to the "
        "end (100%). This explains the ROUGE results above.",
        lambda: positions_verdict(res, m1["position_systems"], m1["position_bins"]),
        lambda: F.model1_positions(res, m1["position_systems"], m1["position_bins"]),
    )
    if m1["example_dialogue"] in res.ids:
        ex = m1["example_dialogue"]
        add(
            "m1-example",
            "Model 1",
            "One dialogue, every system",
            "Importance score of every utterance (columns) by every system (rows) on one test "
            "dialogue; darker = more important. Circles mark the top-3 picks that become the "
            "notes.",
            lambda: example_verdict(res, main, ex),
            lambda: F.model1_example(res, ex),
        )
    add(
        "m1-size",
        "Model 1",
        "Quality against model size",
        "ROUGE-L of the notes against the number of parameters (log scale). Up-left is best: "
        "high quality from a small model.",
        lambda: size_verdict(res, main, th),
        lambda: F.model1_size(res),
    )
    return sections


# ----------------------------------------------------------------------------- HTML

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --line:#e1e0d9; --good:#0ca30c; --mixed:#fab219; --poor:#d03b3b;
  --info:#898781; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 28px; margin: 0 0 4px; }
h2 { font-size: 21px; margin: 40px 0 12px; }
h3 { font-size: 17px; margin: 0 0 6px; }
.sub { color: var(--ink2); margin: 0 0 24px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px; margin: 0 0 20px; }
.what { color: var(--ink2); margin: 0 0 12px; }
.verdict { display:flex; gap:12px; align-items:flex-start; border-left: 4px solid var(--c);
  background: color-mix(in srgb, var(--c) 8%, var(--surface)); padding: 10px 14px;
  border-radius: 0 8px 8px 0; margin: 12px 0 0; }
.verdict p { margin: 0; }
.verdict ul { margin: 6px 0 0; padding-left: 18px; color: var(--ink2); }
.badge { display:inline-flex; align-items:center; gap:6px; font-weight:600; font-size:13px;
  white-space:nowrap; color: var(--ink); }
.badge i { font-style:normal; display:inline-grid; place-items:center; width:20px; height:20px;
  border-radius:50%; background: var(--c); color:#fff; font-size:12px; }
.good { --c: var(--good); } .mixed { --c: var(--mixed); } .poor { --c: var(--poor); }
.info { --c: var(--info); }
table { width:100%; border-collapse: collapse; }
td { padding: 8px 6px; border-top: 1px solid var(--line); vertical-align: top; }
td:first-child { width: 110px; }
td a { color: var(--ink); font-weight: 600; text-decoration: none; }
td a:hover { text-decoration: underline; }
.legend { color: var(--ink2); font-size: 13px; }
code { background: var(--line); padding: 1px 4px; border-radius: 4px; font-size: 13px; }
"""


def badge(level: str) -> str:
    icon, label = LEVELS[level]
    return f'<span class="badge {level}"><i>{icon}</i>{label}</span>'


def render(sections: list[Section]) -> str:
    """The full HTML page: scorecard first, then one card per figure."""
    models = list(dict.fromkeys(s.model for s in sections))
    intro = {
        "Model 2": "restores punctuation and casing in lowercase, unpunctuated text (7.3M "
        "parameters, trained from scratch).",
        "Model 1": "picks the most important utterances of a dialogue to make notes, distilled "
        "from Qwen teachers (17M parameters, trained from scratch).",
    }
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Model Results Report</title><style>{CSS}</style></head><body><main>",
        "<h1>Results report</h1>",
        f"<p class='sub'>Built {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from the saved test "
        "results by <code>python -m src.eval.report</code>. Every verdict is computed from the "
        "numbers with the thresholds in <code>configs/report.yaml</code>. Students and Model 2: "
        "mean ± std over 3 seeds. Hover any chart for exact values.</p>",
        "<div class='card'><h3>Scorecard</h3><p class='legend'>"
        + " · ".join(badge(k) for k in LEVELS)
        + " — context sections explain results rather than judge them.</p><table>",
    ]
    for model in models:
        out.append(f"<tr><td colspan='2'><strong>{model}</strong> {intro.get(model, '')}</td></tr>")
        for s in (s for s in sections if s.model == model):
            out.append(
                f"<tr><td>{badge(s.verdict.level)}</td><td><a href='#{s.key}'>"
                f"{html.escape(s.headline)}</a><br>{html.escape(s.verdict.summary)}</td></tr>"
            )
    out.append("</table></div>")
    first = True
    for model in models:
        out.append(f"<h2>{model}</h2>")
        for s in (s for s in sections if s.model == model):
            fig_html = pio.to_html(
                s.figure,
                full_html=False,
                include_plotlyjs=first,
                config={"displaylogo": False, "responsive": True},
            )
            first = False
            details = "".join(f"<li>{html.escape(d)}</li>" for d in s.verdict.details)
            out += [
                f"<section class='card' id='{s.key}'><h3>{html.escape(s.headline)}</h3>",
                f"<p class='what'><strong>What you are looking at.</strong> "
                f"{html.escape(s.what)}</p>{fig_html}",
                f"<div class='verdict {s.verdict.level}'>{badge(s.verdict.level)}<div>"
                f"<p>{html.escape(s.verdict.summary)}</p>"
                + (f"<ul>{details}</ul>" if details else "")
                + "</div></div></section>",
            ]
    out.append("</main></body></html>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    """Build the report and write it to ``output``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/report.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    sections = build_sections(cfg)
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(sections))
    for s in sections:
        log.info("%-6s %-45s %s", s.verdict.level, s.headline, s.verdict.summary)
    log.info("wrote %s (%d sections)", out, len(sections))


if __name__ == "__main__":
    main()
