"""MLM pretraining: packing, masking, tied head, resumable data order, end-to-end smoke run."""

import json

import keras
import numpy as np
import pytest

from src.model1_scorer import mlm_data, mlm_train
from src.model1_scorer.mlm_model import build_mlm_model, encoder_model
from src.model1_scorer.mlm_train import FIRST_REGULAR_ID, MLMDataset, mask_batch
from src.tokenizer.tokenizer import CLS_ID, MASK_ID, PAD_ID, SEP_ID, load_tokenizer

DATA_CFG = {"mask_prob": 0.15, "mask_token_frac": 0.8, "random_token_frac": 0.1}
MODEL_CFG = {
    "vocab_size": 50,
    "max_len": 12,
    "d_model": 8,
    "num_layers": 1,
    "num_heads": 2,
    "ffn_dim": 16,
    "dropout": 0.0,
    "init_std": 0.02,
    "num_speakers": 8,
}


def test_pack_stream_layout(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    rows = mlm_data.pack_stream(tok, ["hello there", "the dog"], seq_len=4)
    stream = [
        *tok.encode("hello there", add_special_tokens=False).ids,
        SEP_ID,
        *tok.encode("the dog", add_special_tokens=False).ids,
        SEP_ID,
    ]
    assert (rows[:, 0] == CLS_ID).all() and rows.shape[1] == 4
    body = rows[:, 1:].ravel()
    assert list(body[: len(stream)]) == stream and (body[len(stream) :] == PAD_ID).all()


def test_mask_batch_rules():
    rng = np.random.default_rng(0)
    ids = rng.integers(FIRST_REGULAR_ID, 1000, size=(64, 200)).astype(np.int32)
    ids[:, 0], ids[:, 100], ids[:, 150:] = CLS_ID, SEP_ID, PAD_ID  # specials are never masked
    inputs, pos, labels, w = mask_batch(ids, DATA_CFG, 1000, np.random.default_rng(1))
    n_regular = (ids[0] >= FIRST_REGULAR_ID).sum()
    assert pos.shape == (64, 30) and (w.sum(axis=1) == round(0.15 * n_regular)).all()
    chosen = pos[w > 0]
    rows = np.repeat(np.arange(64), w.sum(axis=1).astype(int))
    assert (ids[rows, chosen] >= FIRST_REGULAR_ID).all()
    np.testing.assert_array_equal(labels[w > 0], ids[rows, chosen])
    unchosen = np.ones_like(ids, dtype=bool)
    unchosen[rows, chosen] = False
    np.testing.assert_array_equal(inputs[unchosen], ids[unchosen])
    frac_mask = (inputs[rows, chosen] == MASK_ID).mean()
    frac_same = (inputs[rows, chosen] == ids[rows, chosen]).mean()
    assert 0.75 < frac_mask < 0.85 and 0.05 < frac_same < 0.15
    assert (inputs >= 0).all() and (inputs < 1000).all()


def test_mask_batch_is_deterministic():
    ids = np.arange(13, 13 + 40, dtype=np.int32).reshape(2, 20)
    a = mask_batch(ids, DATA_CFG, 100, np.random.default_rng(5))
    b = mask_batch(ids, DATA_CFG, 100, np.random.default_rng(5))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_resumed_epoch_continues_the_data_order():
    ids = np.arange(13, 13 + 10 * 12, dtype=np.int32).reshape(10, 12)
    spk = np.zeros_like(ids, dtype=np.int8)
    a = MLMDataset(ids, spk, 2, DATA_CFG, 200, seed=3, steps_per_epoch=3)
    first_run = []
    for epoch in range(3):
        a.set_epoch(epoch)
        first_run += [a[i][1] for i in range(len(a))]
    b = MLMDataset(ids, spk, 2, DATA_CFG, 200, seed=3, steps_per_epoch=3)
    b.set_epoch(2)  # as SyncEpoch does after BackupAndRestore
    for i in range(3):
        np.testing.assert_array_equal(b[i][1], first_run[6 + i])
    fixed = MLMDataset(ids, spk, 4, DATA_CFG, 200, seed=3, fixed=True)
    assert len(fixed) == 3
    _, labels, weights = fixed[2]  # 10 rows / 4: the last batch has 2 real rows + 2 padding
    assert len(labels) == 4 and weights[:2].sum() > 0 and (weights[2:] == 0).all()
    np.testing.assert_array_equal(fixed[1][0]["token_ids"], fixed[1][0]["token_ids"])


def test_head_is_tied_and_gathers_positions():
    model = build_mlm_model({**MODEL_CFG, "tie_embeddings": True})
    head_shapes = [tuple(w.shape) for w in model.head.weights]
    assert (50, 8) not in head_shapes and (8, 50) not in head_shapes  # no separate output matrix
    rng = np.random.default_rng(0)
    x = {
        "token_ids": rng.integers(13, 50, (2, 12)).astype("int32"),
        "speaker_ids": rng.integers(0, 9, (2, 12)).astype("int32"),
        "masked_positions": np.array([[3, 7], [0, 11]], dtype="int32"),
    }
    logits = np.asarray(model(x))
    assert logits.shape == (2, 2, 50)
    hidden = np.asarray(model.encoder(x["token_ids"], x["speaker_ids"]))
    h = np.asarray(model.head.norm(model.head.transform(hidden[[0, 0, 1, 1], [3, 7, 0, 11]])))
    emb = np.asarray(model.encoder.embedding.token_embedding.embeddings)
    expected = h @ emb.T + np.asarray(model.head.bias)
    np.testing.assert_allclose(logits.reshape(4, 50), expected, rtol=1e-4, atol=1e-5)


def test_untied_is_rejected():
    with pytest.raises(ValueError, match="tied"):
        build_mlm_model({**MODEL_CFG, "tie_embeddings": False})


def test_save_and_load_mlm_and_encoder(tmp_path):
    model = build_mlm_model(MODEL_CFG)
    x = {
        "token_ids": np.full((1, 12), 20, "int32"),
        "speaker_ids": np.ones((1, 12), "int32"),
        "masked_positions": np.array([[1, 2]], "int32"),
    }
    model.save(tmp_path / "mlm.keras")
    np.testing.assert_allclose(
        keras.saving.load_model(tmp_path / "mlm.keras")(x), model(x), atol=1e-6
    )
    encoder_model(model).save(tmp_path / "encoder.keras")
    enc = keras.saving.load_model(tmp_path / "encoder.keras")
    inputs = {k: x[k] for k in ("token_ids", "speaker_ids")}
    assert enc(inputs).shape == (1, 12, 8)
    np.testing.assert_allclose(
        enc(inputs), model.encoder(x["token_ids"], x["speaker_ids"]), atol=1e-6
    )


def _config(tmp_path, configs_dir, tiny_assets, **training):
    path = tmp_path / "mlm.yaml"
    extra = "".join(f"  {k}: {v}\n" for k, v in training.items())
    path.write_text(
        f"base: {configs_dir / 'mlm_tiny.yaml'}\nname: mlm_test\n"
        f"paths:\n  raw_dir: {tiny_assets['raw']}\n  tokenizer_dir: {tiny_assets['tokenizer']}\n"
        f"  processed_dir: {tmp_path / 'proc'}\n  output_dir: {tmp_path / 'ckpt'}\n"
        f"  results_dir: {tmp_path / 'results'}\n"
        "data:\n  seq_len: 32\n  max_units: 10\nmodel:\n  vocab_size: 400\n  max_len: 32\n"
        f"training:\n  batch_size: 4\n{extra}"
    )
    return path


def test_build_and_pretrain(tmp_path, configs_dir, tiny_assets):
    cfg = _config(tmp_path, configs_dir, tiny_assets, max_steps=6, steps_per_epoch=3)
    mlm_data.main(["--config", str(cfg)])
    proc = tmp_path / "proc"
    assert sorted(p.name for p in proc.iterdir()) == [
        "dialogsum_validation",
        "train",
        "wikitext_validation",
    ]
    meta = json.loads((proc / "train" / "meta.json").read_text())
    assert set(meta["sources"]) == {"wikitext_train", "dialogsum_train"}
    assert meta["sources"]["dialogsum_train"]["dialogues"] == 10  # max_units
    assert meta["sources"]["dialogsum_train"]["copies"] == 5
    wiki_rows = meta["sources"]["wikitext_train"]["rows"]
    assert meta["rows"] == wiki_rows + 5 * 10
    ids = np.load(proc / "train" / "token_ids.npy")
    np.testing.assert_array_equal(
        ids[wiki_rows : wiki_rows + 10], ids[wiki_rows + 10 : wiki_rows + 20]
    )
    speakers = np.load(proc / "train" / "speaker_ids.npy")
    assert speakers.max() == 2 and np.load(proc / "train" / "token_ids.npy").shape[1] == 32

    mlm_train.main(["--config", str(cfg)])
    results = tmp_path / "results" / "mlm_test"
    manifest = json.loads((results / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["plan"]["max_steps"] == 6
    metrics = json.loads((results / "metrics.json").read_text())
    assert set(metrics["held_out"]) == {"wikitext_validation", "dialogsum_validation"}
    assert 0 <= metrics["held_out"]["dialogsum_validation"]["accuracy"] <= 1
    assert len((results / "history.csv").read_text().splitlines()) == 3
    ckpt = tmp_path / "ckpt"
    assert (ckpt / "encoder.keras").exists() and (ckpt / "mlm.keras").exists()
    assert not (ckpt / "backup").exists()
    with pytest.raises(FileExistsError):
        mlm_train.main(["--config", str(cfg)])


def test_throughput_probe_sets_the_budget(tmp_path, configs_dir, tiny_assets):
    cfg = _config(
        tmp_path,
        configs_dir,
        tiny_assets,
        max_steps="null",
        probe_steps=2,
        target_hours=0.0005,
        steps_per_epoch=2,
    )
    mlm_data.main(["--config", str(cfg)])
    from src.config import load_config

    config = load_config(cfg)
    ids, spk = mlm_train.load_set(tmp_path / "proc", "train")
    ds = MLMDataset(ids, spk, 4, config["data"], 400, steps_per_epoch=2)
    plan = mlm_train.plan_steps(config, ds, tmp_path / "out")
    assert plan["source"] == "throughput" and plan["steps_per_second"] > 0
    assert plan["max_steps"] == int(0.0005 * 3600 * plan["steps_per_second"])
    assert mlm_train.plan_steps(config, ds, tmp_path / "out") == plan  # reused on resume


def test_interrupted_run_resumes_with_optimizer_state(tmp_path, configs_dir, tiny_assets):
    cfg = _config(tmp_path, configs_dir, tiny_assets, max_steps=6, steps_per_epoch=3)
    mlm_data.main(["--config", str(cfg)])
    text = cfg.read_text()
    cfg.write_text(text + "max_hours: 0\n")  # time guard stops after the first batch
    mlm_train.main(["--config", str(cfg)])
    results = tmp_path / "results" / "mlm_test"
    assert json.loads((results / "manifest.json").read_text())["status"] == "interrupted"
    ckpt = tmp_path / "ckpt"
    assert (ckpt / "backup").exists() and (ckpt / "encoder.keras").exists()

    cfg.write_text(text)
    mlm_train.main(["--config", str(cfg)])
    manifest = json.loads((results / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["epochs_this_session"] == 1
    model = keras.saving.load_model(ckpt / "mlm.keras")
    assert int(model.optimizer.iterations) == 1 + 3  # 1 step before the cut, 3 after resuming
