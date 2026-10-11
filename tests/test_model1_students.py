"""Model 1 students: arrays and targets, loss, model, inference rule, train -> evaluate."""

import json

import numpy as np
import pytest

from src.data.io import read_jsonl, write_jsonl
from src.model1_scorer import data as m1data
from src.model1_scorer import evaluate, train
from src.model1_scorer.losses import ImportanceLoss
from src.model1_scorer.mlm_model import build_mlm_model, encoder_model
from src.model1_scorer.model import build_student, load_encoder_weights
from src.model1_scorer.notes import grouped, notes_text, render_notes, select_top_k
from src.tokenizer.tokenizer import CLS_ID, load_tokenizer

UTTS = [
    ["Hi, can you send the report?", "Sure, I will send it on Friday.", "Thanks a lot."],
    ["Did you see the game?", "Yes, it was great!", "We decided to buy tickets.", "Cool."],
]
SMALL = {
    "vocab_size": 400,
    "max_len": 32,
    "d_model": 16,
    "num_layers": 1,
    "num_heads": 2,
    "ffn_dim": 32,
    "dropout": 0.0,
    "init_std": 0.02,
    "num_speakers": 8,
}


@pytest.fixture
def raw(tmp_path):
    """DialogSum-style splits with unique IDs and summaries."""
    for split, n in (("train", 12), ("validation", 4), ("test", 4)):
        records = [
            {
                "id": f"{split}_{i}",
                "speakers": [1, 2] * 2,
                "utterances": UTTS[i % 2],
                "summaries": ["#Person2# will send the report on Friday."]
                * (3 if split == "test" else 1),
            }
            for i in range(n)
        ]
        for r in records:
            r["speakers"] = r["speakers"][: len(r["utterances"])]
        write_jsonl(tmp_path / "raw" / "dialogsum" / f"{split}.jsonl", records)
    return tmp_path / "raw"


def write_teacher_labels(labels_dir, raw, skip=()):
    for split in ("train", "validation", "test"):
        rows = []
        for r in read_jsonl(raw / "dialogsum" / f"{split}.jsonl"):
            if r["id"] in skip:
                continue
            n = len(r["utterances"])
            raw_scores = [(7 * j + len(r["id"])) % 11 for j in range(n)]
            tags = [m1data.TAGS[(j + 1) % 4] for j in range(n)]
            rows.append(
                {
                    "id": r["id"],
                    "n": n,
                    "raw_scores": raw_scores,
                    "scores": [s / 10 for s in raw_scores],
                    "tags": tags,
                }
            )
        write_jsonl(labels_dir / f"{split}.jsonl", rows)


# --- arrays and targets -------------------------------------------------------------------


def test_encode_split_and_targets(tmp_path, raw, tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    records = list(read_jsonl(raw / "dialogsum" / "validation.jsonl"))
    arrays, counts = m1data.encode_split(tok, records, max_len=32, max_utterances=3)
    assert arrays["cls_positions"].shape == (4, 3)
    np.testing.assert_array_equal(arrays["n_utterances"], [3, 3, 3, 3])
    assert counts["dropped_utterances"] == 2  # two 4-utterance dialogues capped at 3
    rows = np.arange(4)[:, None]
    assert (arrays["token_ids"][rows, arrays["cls_positions"]] == CLS_ID).all()

    out = tmp_path / "proc"
    out.mkdir()
    for name, array in arrays.items():
        np.save(out / f"{name}.npy", array)
    (out / "ids.json").write_text(json.dumps([r["id"] for r in records]))
    data = m1data.StudentSet.load(out)
    labels = tmp_path / "labels"
    write_teacher_labels(labels, raw, skip={"validation_2"})
    t = m1data.load_targets(labels, "validation", data, "qwen7b")
    np.testing.assert_array_equal(t.labelled, [True, True, False, True])
    assert t.scores[1].tolist() == pytest.approx([0.1, 0.8, 0.4])  # 4th utterance dropped
    assert (t.scores[2] == -1).all() and t.tags[0].tolist() == [1, 2, 3]
    write_jsonl(
        labels / "validation.jsonl",
        [{"id": "validation_0", "labels": [0, 1, 0]}, {"id": "not_encoded", "labels": [1]}],
    )
    oracle = m1data.load_targets(labels, "validation", data, "oracle")
    assert oracle.scores[0].tolist() == [0, 1, 0] and (oracle.tags == -1).all()


# --- loss ---------------------------------------------------------------------------------


def test_importance_loss_hand_checked():
    loss = ImportanceLoss(listwise_lambda=1.0, temperature=2.0, score_scale=10.0)
    target = np.array([[0.2, 0.8, -1.0]], "float32")
    logits = np.array([[0.0, 0.0, 5.0]], "float32")
    p = np.exp(np.array([1.0, 4.0]) - np.log(np.exp(1.0) + np.exp(4.0)))
    kl = float(np.sum(p * (np.log(p) - np.log(0.5))))
    bce = float(np.log(2.0))  # sigmoid(0) = 0.5 for any target
    assert float(loss.call(target, logits)[0]) == pytest.approx(bce + kl, rel=1e-5)
    # Masked utterances do not matter.
    other = logits.copy()
    other[0, 2] = -30.0
    assert float(loss.call(target, other)[0]) == pytest.approx(float(loss.call(target, logits)[0]))
    # The listwise term vanishes when logits = scale * target (+ a constant).
    no_kl = ImportanceLoss(listwise_lambda=1.0, score_scale=10.0)
    aligned = np.array([[2.0 + 3.0, 8.0 + 3.0, 0.0]], "float32")
    only_bce = ImportanceLoss(listwise_lambda=0.0)
    assert float(no_kl.call(target, aligned)[0]) == pytest.approx(
        float(only_bce.call(target, aligned)[0]), abs=1e-5
    )


# --- model and inference rule -------------------------------------------------------------


def test_student_outputs_and_mlm_init(tmp_path):
    student = build_student(SMALL, use_tags=True)
    x = {
        "token_ids": np.full((2, 32), 20, "int32"),
        "speaker_ids": np.ones((2, 32), "int32"),
        "cls_positions": np.array([[0, 4, 9], [0, 6, 0]], "int32"),
    }
    out = student(x)
    assert out["importance"].shape == (2, 3) and out["tags"].shape == (2, 3, 4)
    assert set(build_student(SMALL, use_tags=False)(x)) == {"importance"}

    mlm = build_mlm_model({**SMALL, "tie_embeddings": True})
    encoder_model(mlm).save(tmp_path / "encoder.keras")
    load_encoder_weights(student, tmp_path / "encoder.keras")
    for a, b in zip(
        student.get_layer("encoder").get_weights(), mlm.encoder.get_weights(), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="mismatch"):
        load_encoder_weights(
            build_student({**SMALL, "d_model": 8}, True), tmp_path / "encoder.keras"
        )


def test_inference_rule():
    assert select_top_k([0.1, 0.9, 0.5, 0.9, 0.2], 3) == [1, 2, 3]
    assert select_top_k([1, 1, 1, 1], 3) == [0, 1, 2]  # ties -> earlier
    assert select_top_k([0.4], 3) == [0]
    utts = ["a", "b", "c", "d"]
    tags = ["action_item", "none", "decision", "action_item"]
    assert grouped([0, 2, 3], tags) == [("decision", [2]), ("action_item", [0, 3])]
    assert notes_text(utts, [0, 2, 3], tags) == "c a d"
    assert notes_text(utts, [3, 0]) == "a d"
    md = render_notes(utts, [1, 2, 1, 2], [0, 2], tags)
    assert md.splitlines() == [
        "**Decisions**",
        "- #Person1#: c",
        "**Action items**",
        "- #Person1#: a",
    ]


# --- train -> evaluate --------------------------------------------------------------------


def _student_config(tmp_path, configs_dir, tiny_assets, raw, name, **extra):
    path = tmp_path / f"{name}.yaml"
    lines = [
        f"base: {configs_dir / 'model1_student_tiny.yaml'}",
        f"name: {name}",
        "paths:",
        f"  raw_dir: {raw}",
        f"  tokenizer_dir: {tiny_assets['tokenizer']}",
        f"  processed_dir: {tmp_path / 'proc'}",
        f"  labels_dir: {extra.pop('labels_dir')}",
        f"  output_dir: {tmp_path / 'ckpt' / name}",
        f"  results_dir: {tmp_path / 'results'}",
        f"  init_checkpoint: {extra.pop('init_checkpoint', 'null')}",
        "model:\n  vocab_size: 400\n  max_len: 32\n  d_model: 16\n  ffn_dim: 32",
        "training:\n  batch_size: 4\n  epochs: 2\n  max_steps: 4",
        *extra.pop("lines", []),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_train_students_and_evaluate(tmp_path, configs_dir, tiny_assets, raw):
    labels = tmp_path / "labels"
    write_teacher_labels(labels / "qwen7b", raw, skip={"train_3"})
    write_teacher_labels(labels / "qwen1_5b", raw, skip={"test_1"})
    oracle = [
        {"id": r["id"], "labels": [0, 1, 0, 0][: len(r["utterances"])], "selected": [1]}
        for r in read_jsonl(raw / "dialogsum" / "test.jsonl")
    ]
    write_jsonl(labels / "oracle" / "test.jsonl", oracle)
    mlm = build_mlm_model({**SMALL, "tie_embeddings": True})
    encoder_model(mlm).save(tmp_path / "encoder.keras")

    cfg_a = _student_config(
        tmp_path,
        configs_dir,
        tiny_assets,
        raw,
        "student_A",
        labels_dir=labels / "qwen7b",
        init_checkpoint=tmp_path / "encoder.keras",
        lines=["init: mlm", "targets:\n  source: qwen7b\n  use_tags: true"],
    )
    m1data.main(["--config", str(cfg_a)])
    train.main(["--config", str(cfg_a), "--all-seeds"])
    res = tmp_path / "results" / "student_A" / "seed_0"
    manifest = json.loads((res / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["train_dialogues"] == 11  # train_3 skipped
    preds = list(read_jsonl(res / "test_predictions.jsonl"))
    assert len(preds) == 4 and all(len(p["scores"]) == len(p["tags"]) for p in preds)
    assert all(0 <= s <= 1 for p in preds for s in p["scores"])

    eval_cfg = tmp_path / "eval.yaml"
    eval_cfg.write_text(
        f"base: {configs_dir / 'model1_eval.yaml'}\n"
        "students:\n"
        f"  student_A: {tmp_path / 'ckpt' / 'student_A'}\n"
        "bertscore:\n  enabled: false\nlatency_dialogues: 2\n"
        f"paths:\n  raw_dir: {raw}\n  labels_dir: {labels}\n  processed_dir: {tmp_path / 'proc'}\n"
        f"  results_dir: {tmp_path / 'results'}\n"
    )
    evaluate.main(["--config", str(eval_cfg)])
    table = (tmp_path / "results" / "tables" / "model1.md").read_text()
    assert "3 test dialogues" in table  # test_1 is not labelled by the 1.5B teacher
    for name in ("Lead-3", "Oracle", "Teacher qwen7b", "Teacher qwen1_5b", "student_A"):
        assert f"| {name} |" in table
    teacher = json.loads((tmp_path / "results" / "model1_eval" / "teacher_qwen7b.json").read_text())
    ref = teacher["runs"][0]
    assert ref["retention"] == pytest.approx(1.0) and ref["spearman"] == pytest.approx(1.0)
    assert ref["jaccard"] == 1.0 and ref["tags"]["decision"]["f1"] == pytest.approx(1.0)
    student = json.loads((tmp_path / "results" / "model1_eval" / "student_a.json").read_text())
    assert student["latency"] != "–" and 0 <= student["runs"][0]["rouge1"] <= 1


# --- length buckets -----------------------------------------------------------------------


def _student_set(lengths, width=32):
    """Dialogues of the given token lengths: [CLS] at 0 and at the middle, padding after."""
    n = len(lengths)
    ids = np.zeros((n, width), "int32")
    for i, k in enumerate(lengths):
        ids[i, :k] = 20 + (np.arange(k) % 7)
        ids[i, 0] = ids[i, k // 2] = CLS_ID
    cls = np.array([[0, k // 2] for k in lengths], "int32")
    return m1data.StudentSet(
        ids=[f"d{i}" for i in range(n)],
        token_ids=ids,
        speaker_ids=(ids > 0).astype("int8"),
        cls_positions=cls,
        n_utterances=np.full(n, 2, "int32"),
    )


def test_length_helpers():
    data = _student_set([5, 32, 9, 17])
    np.testing.assert_array_equal(train.row_lengths(data.token_ids), [5, 32, 9, 17])
    assert [train.bucket_length(n, [8, 16, 24], 32) for n in (3, 8, 9, 25)] == [8, 8, 16, 32]
    assert train.bucket_length(30, [64, 128], 32) == 32  # never wider than the arrays

    lengths = np.array([7, 1, 9, 3, 5, 2, 8, 4, 6, 0])
    rows = np.arange(10)
    ev = train.length_batches(rows, lengths, 4)
    assert [b.tolist() for b in ev] == [[9, 1, 5, 3], [7, 4, 8, 0], [6, 2]]
    a = train.length_batches(rows, lengths, 2, np.random.default_rng(0), pool_batches=2)
    b = train.length_batches(rows, lengths, 2, np.random.default_rng(0), pool_batches=2)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]  # deterministic per rng
    assert sorted(np.concatenate(a).tolist()) == rows.tolist()  # every row exactly once
    for batch in a:  # each batch comes from one sorted pool: lengths are adjacent in it
        assert len(batch) == 2


def test_bucketed_dataset_and_predictions_match_unbucketed():
    lengths = [5, 30, 9, 17, 6, 12, 28, 3]
    data = _student_set(lengths)
    targets = m1data.Targets(
        scores=np.tile(np.array([[0.2, 0.9]], "float32"), (8, 1)),
        tags=np.ones((8, 2), "int8"),
        labelled=np.ones(8, bool),
    )
    ds = train.StudentDataset(
        data, targets, np.arange(8), 2, True, seed=0, buckets=[8, 16, 24], pool_batches=2
    )
    assert len(ds) == 4
    widths, kept = [], []
    for i in range(len(ds)):
        x, y, sw = ds[i]
        width = x["token_ids"].shape[1]
        assert width in (8, 16, 24, 32) and x["speaker_ids"].shape == x["token_ids"].shape
        assert y["importance"].shape == (2, 2) and sw["tags"].shape == (2, 2)
        widths.append(width)
        kept += train.row_lengths(x["token_ids"]).tolist()
    assert min(widths) < 32  # some batches are cut ...
    assert sorted(kept) == sorted(lengths)  # ... but only padding is removed

    model = build_student(SMALL, use_tags=True)
    rows = np.array([3, 0, 6, 1, 7])
    plain = train.predict_utterances(model, data, rows, 2)
    bucketed = train.predict_utterances(model, data, rows, 2, buckets=[8, 16, 24])
    for p, q in zip(plain, bucketed, strict=True):
        assert p["tags"] == q["tags"]
        np.testing.assert_allclose(p["scores"], q["scores"], atol=1e-5)
