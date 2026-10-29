"""Model 1 v3 student: model, losses, batching and a tiny end-to-end run (PLAN.md 10.4-10.5)."""

import json

import keras
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from keras import ops

from src.config import deep_merge, load_config
from src.model1_notes import train
from src.model1_notes.data import REASONS, SPLITS, NotesSet, build_split, sentence_rows
from src.model1_notes.encoder_teachers import write_logits
from src.model1_notes.losses import NotesImportanceLoss, ZoneLoss
from src.model1_notes.model import NEG, build_notes_model, load_sentence_encoder
from src.model1_notes.tiny_corpus import write_tiny_corpus
from src.model1_scorer.mlm_model import build_mlm_model, encoder_model
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID, load_tokenizer

SMALL = {
    "vocab_size": 50,
    "max_len": 8,
    "d_model": 16,
    "num_layers": 1,
    "meeting_layers": 1,
    "num_heads": 2,
    "ffn_dim": 32,
    "dropout": 0.0,
    "init_std": 0.02,
    "num_speakers": 8,
    "num_zones": 5,
    "num_reasons": 9,
    "num_span_tags": 5,
}


def _inputs(n_real: int, slots: int, t: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = np.full((1, slots, t), PAD_ID, dtype=np.int32)
    speakers = np.zeros((1, slots), dtype=np.int32)
    for j in range(n_real):
        tokens[0, j, :4] = [CLS_ID, *rng.integers(13, 50, 2), SEP_ID]
        speakers[0, j] = 1 + j % 2
    return {"token_ids": tokens, "speaker_ids": speakers}


def test_output_shapes_and_padding_invariance():
    model = build_notes_model(SMALL, max_sentences=16, sentence_tokens=8)
    out = model(_inputs(3, 4))
    assert tuple(out["importance"].shape) == (1, 4)
    assert tuple(out["zone"].shape) == (1, 4, 5) and tuple(out["reason"].shape) == (1, 4, 9)
    assert tuple(out["depends_on"].shape) == (1, 4, 5)
    assert tuple(out["spans"].shape) == (1, 4, 8, 5)
    wider = model(_inputs(3, 8))  # more empty slots must not change real sentences
    np.testing.assert_allclose(
        np.asarray(out["importance"])[0, :3], np.asarray(wider["importance"])[0, :3], atol=1e-5
    )


def test_pointer_only_points_to_earlier_real_sentences():
    model = build_notes_model(SMALL, max_sentences=16, sentence_tokens=8)
    logits = np.asarray(model(_inputs(3, 4))["depends_on"])[0]
    assert (logits[:, :4][np.triu_indices(4)] <= NEG / 2).all()  # j >= i
    assert (logits[:, 3] <= NEG / 2).all()  # nobody points to the empty slot
    assert np.isfinite(logits[:, 4]).all() and (np.abs(logits[:, 4]) < 1e3).all()
    assert logits[2, 0] > NEG / 2 and logits[2, 1] > NEG / 2


def test_save_load_round_trip(tmp_path):
    model = build_notes_model(SMALL, max_sentences=16, sentence_tokens=8)
    x = _inputs(3, 4)
    model.save(tmp_path / "m.keras")
    loaded = keras.saving.load_model(tmp_path / "m.keras", compile=False)
    for key, value in model(x).items():
        np.testing.assert_allclose(np.asarray(value), np.asarray(loaded(x)[key]), atol=1e-6)


def test_load_sentence_encoder_from_mlm(tmp_path):
    mlm_cfg = {k: SMALL[k] for k in SMALL if k not in ("meeting_layers",)}
    mlm = build_mlm_model({**mlm_cfg, "tie_embeddings": True})
    encoder_model(mlm).save(tmp_path / "encoder.keras")
    model = build_notes_model(SMALL, max_sentences=16, sentence_tokens=8)
    load_sentence_encoder(model, tmp_path / "encoder.keras")
    for a, b in zip(
        model.get_layer("encoder").get_weights(), mlm.encoder.get_weights(), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="mismatch"):
        other = build_notes_model({**SMALL, "d_model": 8}, 16, 8)
        load_sentence_encoder(other, tmp_path / "encoder.keras")


def test_importance_loss_terms():
    logit = np.array([[2.0, -1.0]], dtype=np.float32)
    gold = np.array([[[1, 0.5, 0.3, 1, 0.0, 0], [0, 1.0, 0.0, 0, 0.0, 0]]], dtype=np.float32)
    bce = ops.binary_crossentropy(np.array([[1.0, 0.0]]), logit, from_logits=True)
    expected = np.asarray(bce) * np.array([[0.5, 1.0]])
    gold_only = NotesImportanceLoss(1.0, 0.0, 0.0, reduction=None)(gold, logit)
    np.testing.assert_allclose(np.asarray(gold_only), expected, rtol=1e-5)
    with_qwen = np.asarray(NotesImportanceLoss(1.0, 1.0, 0.0, reduction=None)(gold, logit))
    assert with_qwen[0, 0] > expected[0, 0]  # Qwen (p = 0.3) disagrees with logit 2
    np.testing.assert_allclose(with_qwen[0, 1], expected[0, 1], rtol=1e-5)  # no Qwen label
    agree = gold.copy()
    agree[0, 0, 4:] = [2.0, 1.0]  # RoBERTa logit equals the student's: KL = 0
    with_rob = NotesImportanceLoss(1.0, 0.0, 1.0, reduction=None)(agree, logit)
    np.testing.assert_allclose(np.asarray(with_rob), expected, rtol=1e-4, atol=1e-6)


def test_zone_loss_masks_context_and_closing_for_ami():
    logits = np.array([[[5.0, 0.0, 0.0, 0.0, 5.0]]], dtype=np.float32)
    synthetic = ZoneLoss(reduction=None)(np.array([[[1, 0]]]), logits)
    ami = ZoneLoss(reduction=None)(np.array([[[1, 1]]]), logits)
    assert float(np.asarray(ami)[0, 0]) < 1.2 < float(np.asarray(synthetic)[0, 0])


def test_depends_targets():
    rows = np.array([[10, 11, 12, -1]])
    depends = np.full(20, -2, dtype=np.int32)
    depends[11], depends[12], depends[10] = 10, 5, -1  # 12 -> outside the window
    labels, weights = train.depends_targets(rows, depends)
    assert labels.tolist() == [[4, 0, 4, 4]] and weights.tolist() == [[1, 1, 1, 0]]


def test_tune_threshold_and_f1():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    threshold, f1 = train.tune_threshold(truth, np.array([0.9, 0.6, 0.4, 0.1]))
    assert f1 == 1.0 and 0.4 < threshold <= 0.6
    assert train.f1_score(truth, np.zeros(4, dtype=bool)) == 0.0


def write_fake_teachers(corpus, teacher_dir, split):
    """Qwen and RoBERTa tables in the Notebook 1 export format (every other meeting labelled)."""
    rows = []
    for layer in ("synthetic", "ami"):
        rows += pq.read_table(
            corpus / "data" / layer / "sentences" / f"{split}.parquet"
        ).to_pylist()
    rng = np.random.default_rng(0)
    notes = sorted({r["note_id"] for r in rows})[::2]
    qwen = [
        {
            "note_id": r["note_id"],
            "sentence_id": r["sentence_id"],
            "score": float(rng.random()),
            "raw_score": 1,
            "reason": REASONS[r["sentence_id"] % len(REASONS)],
            "depends_on": r["sentence_id"] - 1,
        }
        for r in rows
        if r["note_id"] in notes
    ]
    (teacher_dir / "qwen").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(qwen), teacher_dir / "qwen" / f"{split}.parquet")
    logits = [
        {"note_id": r["note_id"], "sentence_id": r["sentence_id"], "logit": float(rng.normal())}
        for r in rows
    ]
    write_logits(teacher_dir / "roberta" / f"{split}.parquet", logits)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, tiny_assets):
    root = tmp_path_factory.mktemp("v3")
    write_tiny_corpus(root / "corpus", meetings_per_split=6)
    cfg = load_config("configs/model1_v3_tiny.yaml")
    cfg = deep_merge(
        cfg,
        {
            "paths": {
                "corpus_dir": str(root / "corpus"),
                "processed_dir": str(root / "processed"),
                "teacher_dir": str(root / "teacher"),
                "output_dir": str(root / "ckpt"),
                "results_dir": str(root / "results"),
            },
            "training": {"max_steps": 4, "sentences_per_batch": 64},
            "max_hours": 1.0,  # the time guard is tested elsewhere; CI load must not trip it
        },
    )
    tokenizer = load_tokenizer(tiny_assets["tokenizer"])
    for split in SPLITS:
        build_split(tokenizer, cfg, split, root / "processed" / split)
        write_fake_teachers(root / "corpus", root / "teacher", split)
    return root, cfg


def test_batches_cover_windows_and_oversample(tiny_run):
    root, cfg = tiny_run
    data = NotesSet.load(root / "processed" / "train")
    windows = train.train_windows(data, 2)
    hn = (np.asarray(data["source"])[np.asarray(data["window_note"])] == 1).sum()
    assert len(windows) == data.num_windows + hn
    batches = train.window_batches(data, windows, cfg["data"]["sentence_buckets"], 64)
    assert sorted(np.concatenate([b for b, _ in batches]).tolist()) == sorted(windows.tolist())
    for ws, bucket in batches:
        assert (np.asarray(data["window_length"])[ws] <= bucket).all()


def test_batch_arrays_targets(tiny_run):
    root, cfg = tiny_run
    data = NotesSet.load(root / "processed" / "train")
    teacher = train.align_teacher(data, root / "teacher", "train", ["qwen", "roberta"])
    x, y, sw = train.batch_arrays(data, teacher, np.array([0, 1]), 32, n_real=1)
    assert x["token_ids"].shape == (2, 32, 16) and y["importance"].shape == (2, 32, 6)
    assert sw["importance"][1].sum() == 0  # repeated padding window
    mask = sw["importance"][0] > 0
    assert (y["importance"][0, mask, 3] == 1).all() and (y["importance"][0, ~mask] == 0).all()
    assert (sw["spans"][0][~mask] == 0).all()


@pytest.mark.parametrize("name", ["E1", "E4"])
def test_tiny_training_end_to_end(tiny_run, name):
    root, cfg = tiny_run
    variant = load_config(f"configs/model1_v3_{name}.yaml")
    cfg = deep_merge(cfg, {"teachers": variant["teachers"], "loss": variant["loss"]})
    cfg = {**cfg, "name": f"tiny_{name}"}
    manifest = train.train_seed(cfg, seed=0)
    assert manifest["status"] == "complete"
    results = root / "results" / f"tiny_{name}" / "seed_0"
    metrics = json.loads((results / "metrics.json").read_text())
    assert 0.05 <= metrics["threshold"] <= 0.95
    assert {"all", "synthetic", "ami"} <= set(metrics["test"])
    lines = (results / "test_predictions.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert len(first["importance"]) == len(first["zone"]) == len(first["sentence_ids"])
    with pytest.raises(FileExistsError):
        train.train_seed(cfg, seed=0)


def test_align_teacher(tiny_run):
    root, _ = tiny_run
    data = NotesSet.load(root / "processed" / "validation")
    none = train.align_teacher(data, None, "validation", ["qwen", "roberta"])
    assert (none["qwen_score"] == -1).all() and np.isnan(none["roberta_logit"]).all()
    got = train.align_teacher(data, root / "teacher", "validation", ["qwen", "roberta"])
    labelled = got["qwen_score"] >= 0
    assert 0 < labelled.sum() < len(labelled)  # every other meeting
    assert not np.isnan(got["roberta_logit"]).any()
    rows = sentence_rows(data)
    first = data.note_ids[0]
    assert got["qwen_depends"][rows[(first, 0)]] == -1  # depends_on -1 -> none
    assert got["qwen_depends"][rows[(first, 2)]] == rows[(first, 1)]
    only_rob = train.align_teacher(data, root / "teacher", "validation", ["roberta"])
    assert (only_rob["qwen_score"] == -1).all()


@pytest.fixture(scope="module")
def trained(tiny_run):
    """An E4-style tiny run whose outputs the evaluation and the notes renderer read."""
    root, cfg = tiny_run
    variant = load_config("configs/model1_v3_E4.yaml")
    cfg = deep_merge(cfg, {"teachers": variant["teachers"], "loss": variant["loss"]})
    cfg = deep_merge(
        cfg, {"name": "eval_E4", "paths": {"output_dir": str(root / "ckpt" / "eval_E4")}}
    )
    train.train_seed(cfg, seed=0)
    return root, cfg


def test_metrics_include_spans_and_zones(trained):
    root, _ = trained
    metrics = json.loads((root / "results" / "eval_E4" / "seed_0" / "metrics.json").read_text())
    test = metrics["test"]
    assert test["spans"]["gold_spans"] > 0 and 0 <= test["spans"]["span_f1"] <= 1
    assert {"decision_f1", "action_f1", "pr_auc", "hard_negative_fp"} <= set(test["synthetic"])


def test_evaluate_table(trained, tmp_path):
    from src.model1_notes import evaluate

    root, cfg = trained
    write_logits(
        root / "teacher" / "distilbert" / "test.parquet",
        pq.read_table(root / "teacher" / "roberta" / "test.parquet").to_pylist(),
    )
    write_logits(
        root / "teacher" / "distilbert" / "validation.parquet",
        pq.read_table(root / "teacher" / "roberta" / "validation.parquet").to_pylist(),
    )
    eval_cfg = load_config("configs/model1_v3_eval.yaml")
    eval_cfg["baselines"] = load_config("configs/model1_v3.yaml")["baselines"]
    eval_cfg["paths"] = {
        "corpus_dir": str(root / "corpus"),
        "teacher_dir": str(root / "teacher"),
        "results_dir": str(root / "results"),
        "checkpoints_dir": str(root / "ckpt"),
        "tables_dir": str(tmp_path / "tables"),
    }
    eval_cfg["students"] = ["eval_E4", "missing_student"]
    eval_cfg["latency"] = {"sentences": [8, 40], "repeats": 1, "sentence_tokens": 16}
    report = evaluate.evaluate(eval_cfg, latency=True)
    assert {"Lead-k", "Keyword rule", "Qwen3-0.6B (teacher)", "eval_E4"} <= set(report)
    assert "missing_student" not in report
    assert (
        report["Lead-k"]["ami"]["sentences"] > 0 and report["Lead-k"]["rouge"]["notes_rouge1"] > 0
    )
    assert report["Qwen3-0.6B (teacher)"]["rouge"]["decisions_rougeL"] >= 0
    seed = report["eval_E4"]["seeds"][0]
    assert set(seed["latency_ms"]) == {"8", "40"}
    md = (tmp_path / "tables" / "model1_v3.md").read_text()
    assert "| eval_E4 |" in md and "AMI imp F1" in md


def test_notes_from_transcript(trained, tiny_assets):
    from src.model1_notes.notes import notes_from_transcript, parse_transcript, render

    root, cfg = trained
    model_dir = root / "ckpt" / "eval_E4" / "seed_0"
    (model_dir / "metrics.json").write_text(json.dumps({"threshold": 0.0}))  # keep everything
    text = "\n".join(
        [
            "Ana: Thanks for joining.",
            "Ben: Ben will fix the rig by Friday.",
            "Ana: Okay, we hold the release.",
            "no speaker line",
        ]
        * 12
    )
    tokenizer = load_tokenizer(tiny_assets["tokenizer"])
    md = notes_from_transcript(text, model_dir, tokenizer, cfg)
    kept = [line for line in md.splitlines() if line.startswith("- ") and line != "- (none)"]
    assert md.count("## ") == 3 and len(kept) == 48  # threshold 0: every sentence is kept
    assert parse_transcript("A: x\n\nno colon")[0] == ["A", "?"]
    assert "- (none)" in render([])


def test_bucket_batch_sizes_keep_a_sentence_budget():
    assert train.bucket_batch_sizes([32, 64, 128, 256], 2048) == {32: 64, 64: 32, 128: 16, 256: 8}
    with pytest.raises(ValueError):
        train.bucket_batch_sizes([32, 256], 1000)
