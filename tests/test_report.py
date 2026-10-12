"""Tests for the results report: verdict rules and an end-to-end build on synthetic results."""

import json

import numpy as np
import yaml

from src.data.io import write_jsonl
from src.eval.report import (
    converged,
    main,
    model2_noise_verdict,
    model2_per_class_verdict,
)
from src.eval.results_data import display_name, parse_params

TAGS = ["none", "key_point", "decision", "action_item"]
THRESHOLDS = {
    "retention_target": 0.9,
    "good_f1": 0.75,
    "poor_f1": 0.5,
    "min_support": 50,
    "good_spearman": 0.6,
    "poor_spearman": 0.3,
    "noise_drop_ok": 0.2,
    "plateau_epochs": 3,
    "plateau_gain": 0.005,
    "unstable_drop": 0.05,
}


def _report(f1, support=1000):
    pc = {
        n: {"f1": f1, "support": support}
        for n in ("comma", "period", "question", "capitalised", "upper")
    }
    return {
        "punct": {"macro_f1": f1, "per_class": pc, "n": 10 * support},
        "casing": {"accuracy": f1, "per_class": pc, "n": 10 * support},
    }


def _model2_seeds(clean, noisy, baseline=0.2, support=1000):
    seeds = {}
    for seed in (0, 1):
        test = {
            f"test_{d}_noise{int(n * 100):02d}": {
                "noise": n,
                "baseline": _report(baseline, support),
                "model": _report((clean if n == 0 else noisy) + 0.001 * seed, support),
            }
            for d in ("wikitext", "dialogsum")
            for n in (0.0, 0.2)
        }
        seeds[seed] = {"seed": seed, "params": 7_000_000, "validation": _report(0.7), "test": test}
    return seeds


def test_display_name_and_params():
    assert display_name("Teacher qwen1_5b") == "Teacher 1.5B"
    assert display_name("model1_student_D_noMLM") == "Student D (no MLM)"
    assert parse_params("1.5B") == 1.5e9
    assert parse_params("17M") == 17e6


def test_noise_verdict_levels():
    assert model2_noise_verdict(_model2_seeds(0.8, 0.7), THRESHOLDS).level == "good"
    assert model2_noise_verdict(_model2_seeds(0.8, 0.5), THRESHOLDS).level == "mixed"  # -37%
    assert model2_noise_verdict(_model2_seeds(0.8, 0.1), THRESHOLDS).level == "poor"  # < baseline


def test_per_class_verdict_skips_rare_classes():
    rare = model2_per_class_verdict(_model2_seeds(0.3, 0.3, support=10), 0.0, THRESHOLDS)
    assert rare.level == "good"  # nothing judged: every class is below min_support
    assert any("Too few examples" in d for d in rare.details)
    weak = model2_per_class_verdict(_model2_seeds(0.3, 0.3), 0.0, THRESHOLDS)
    assert weak.level == "poor"


def test_converged():
    flat = np.array([[0.5, 0], [0.7, 0], [0.701, 0], [0.702, 0], [0.703, 0]])
    rising = np.array([[0.5, 0], [0.6, 0], [0.65, 0], [0.7, 0], [0.75, 0]])
    assert converged(flat, THRESHOLDS)[0]
    assert not converged(rising, THRESHOLDS)[0]


def _history(path, columns, epochs=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [",".join(["epoch", *columns])]
    rows += [",".join([str(e), *[str(0.1 * (e + 1))] * len(columns)]) for e in range(epochs)]
    path.write_text("\n".join(rows) + "\n")


def _write_results(root):
    """Model 2 seeds, MLM run, DialogSum test, teacher / oracle labels and two students."""
    results, labels, raw = root / "results", root / "labels", root / "raw"
    for seed, metrics in _model2_seeds(0.8, 0.7).items():
        out = results / "model2_cleanup" / f"seed_{seed}"
        out.mkdir(parents=True)
        (out / "metrics.json").write_text(json.dumps(metrics))
        _history(
            out / "history.csv", ["loss", "val_loss", "val_punct_macro_f1", "val_casing_macro_f1"]
        )
    _history(results / "model1_mlm" / "history.csv", ["acc", "loss", "val_acc", "val_loss"])
    held = {"wikitext_validation": {"accuracy": 0.5}, "dialogsum_validation": {"accuracy": 0.6}}
    (results / "model1_mlm" / "metrics.json").write_text(json.dumps({"held_out": held}))

    rng = np.random.default_rng(0)
    lengths = [4, 6, 5, 7]
    dialogues = [
        {
            "id": f"test_{i}",
            "speakers": [1 + j % 2 for j in range(n)],
            "utterances": [f"utterance {j} of dialogue {i} about topic {j % 3}" for j in range(n)],
            "summaries": [f"dialogue {i} about topic 1", f"topic 2 in dialogue {i}"],
        }
        for i, n in enumerate(lengths)
    ]
    write_jsonl(raw / "dialogsum" / "test.jsonl", dialogues)
    for teacher in ("qwen7b", "qwen1_5b"):
        write_jsonl(
            labels / teacher / "test.jsonl",
            [
                {
                    "id": d["id"],
                    "scores": [float(x) for x in rng.random(n)],
                    "tags": [TAGS[j % 4] for j in range(n)],
                }
                for d, n in zip(dialogues, lengths, strict=True)
            ],
        )
    write_jsonl(
        labels / "oracle" / "test.jsonl",
        [
            {"id": d["id"], "labels": [1, 1] + [0] * (n - 2), "selected": [0, 1]}
            for d, n in zip(dialogues, lengths, strict=True)
        ],
    )
    for name, tags in (("model1_student_A", True), ("model1_student_C", False)):
        for seed in (0, 1):
            out = results / name / f"seed_{seed}"
            preds = []
            for d, n in zip(dialogues, lengths, strict=True):
                entry = {"id": d["id"], "scores": [float(x) for x in rng.random(n)]}
                if tags:
                    entry["tags"] = [TAGS[(j + seed) % 4] for j in range(n)]
                preds.append(entry)
            write_jsonl(out / "test_predictions.jsonl", preds)
            (out / "metrics.json").write_text(json.dumps({"seed": seed, "params": 17_000_000}))
            _history(out / "history.csv", ["loss", "val_spearman"])

    eval_cfg = {
        "name": "model1_eval",
        "seed": 0,
        "top_k": 3,
        "use_stemmer": True,
        "reference_teacher": "qwen7b",
        "teachers": ["qwen7b", "qwen1_5b"],
        "teacher_params": {"qwen7b": "7B", "qwen1_5b": "1.5B"},
        "students": {"model1_student_A": "none", "model1_student_C": "none"},
        "bertscore": {"enabled": True, "model_type": "unused", "batch_size": 1},
        "paths": {"raw_dir": str(raw), "labels_dir": str(labels), "results_dir": str(results)},
    }
    (root / "model1_eval.yaml").write_text(yaml.safe_dump(eval_cfg))
    cfg = {
        "name": "report",
        "seed": 0,
        "output": str(root / "report" / "index.html"),
        "model2": {"exp": "model2_cleanup", "per_class_noise": 0.0},
        "model1": {
            "eval_config": str(root / "model1_eval.yaml"),
            "mlm_dir": str(results / "model1_mlm"),
            "main_student": "model1_student_A",
            "ablation": {"with_mlm": "model1_student_A", "without_mlm": "missing"},
            "example_dialogue": "test_1",
            "position_bins": 4,
            "position_systems": ["Lead-3", "Oracle", "model1_student_A", "missing_system"],
        },
        "verdicts": THRESHOLDS,
    }
    (root / "report.yaml").write_text(yaml.safe_dump(cfg))


def test_main_builds_every_section(tmp_path):
    """Without model1_eval JSONs the report recomputes ROUGE (no BERTScore) and builds all."""
    _write_results(tmp_path)
    main(["--config", str(tmp_path / "report.yaml")])
    page = (tmp_path / "report" / "index.html").read_text()
    for key in (
        "m2-noise", "m2-classes", "m2-training", "m1-mlm", "m1-training", "m1-quality",
        "m1-agreement", "m1-spread", "m1-tags", "m1-positions", "m1-example", "m1-size",
    ):  # fmt: skip
        assert f"id='{key}'" in page
    assert "What you are looking at." in page
    assert "BERTScore is not shown yet" in page
    assert page.count("plotly.js") >= 1 and page.count("<script") >= 12
