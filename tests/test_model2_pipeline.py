"""End-to-end Model 2 smoke test on tiny data: build arrays, train a few steps, evaluate."""

import json

import numpy as np

from src.config import load_config
from src.layers.losses import IGNORE
from src.model2_cleanup import build_data, train
from src.model2_cleanup.dataset import ArraySet, CleanupDataset
from src.model2_cleanup.evaluate import rule_baseline, score
from src.model2_cleanup.examples import encode_windows
from src.model2_cleanup.labels import CAPITALISED, LOWER, PERIOD, label_text
from src.tokenizer.tokenizer import load_tokenizer


def _config(tmp_path, configs_dir, tiny_assets):
    path = tmp_path / "m2.yaml"
    path.write_text(
        f"base: {configs_dir / 'model2_tiny.yaml'}\nname: m2_test\n"
        f"paths:\n  raw_dir: {tiny_assets['raw']}\n  tokenizer_dir: {tiny_assets['tokenizer']}\n"
        f"  processed_dir: {tmp_path / 'proc'}\n  output_dir: {tmp_path / 'ckpt'}\n"
        f"  results_dir: {tmp_path / 'results'}\n"
        "model:\n  vocab_size: 400\n"
        "training:\n  max_steps: 6\n  steps_per_epoch: 3\n  epochs: 2\n  batch_size: 4\n"
    )
    return path


def test_build_train_evaluate(tmp_path, configs_dir, tiny_assets):
    cfg_path = _config(tmp_path, configs_dir, tiny_assets)
    build_data.main(["--config", str(cfg_path)])
    proc = tmp_path / "proc"
    names = sorted(p.name for p in proc.iterdir() if p.is_dir())
    levels = ("00", "05", "10", "20")
    tests = [f"test_{d}_noise{n}" for d in ("wikitext", "dialogsum") for n in levels]
    assert names == sorted(["train", "validation_wikitext", "validation_dialogsum", *tests])
    train_set = ArraySet.load(proc / "train")
    assert train_set.token_ids.dtype == np.int32 and train_set.punct.dtype == np.int8
    assert train_set.token_ids.shape[1] == 32
    meta = json.loads((proc / "meta.json").read_text())
    assert len(meta["punct_class_weights"]) == 4
    clean = ArraySet.load(proc / "test_wikitext_noise00")
    noisy = ArraySet.load(proc / "test_wikitext_noise20")
    assert clean.meta["noise_total"] == 0 and noisy.meta["noise_total"] == 0.2
    assert not np.array_equal(clean.token_ids, noisy.token_ids[: len(clean)])

    # Same data seed -> identical arrays.
    build_data.main(["--config", str(cfg_path), "--sets", "test"])
    again = ArraySet.load(proc / "test_wikitext_noise20")
    np.testing.assert_array_equal(again.token_ids, noisy.token_ids)

    train.main(["--config", str(cfg_path), "--all-seeds"])
    results = tmp_path / "results" / "m2_test" / "seed_0"
    metrics = json.loads((results / "metrics.json").read_text())
    manifest = json.loads((results / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["config"]["name"] == "m2_test"
    assert set(metrics["test"]) == {n for n in names if n.startswith("test_")}
    entry = metrics["test"]["test_dialogsum_noise00"]
    assert 0 <= entry["model"]["punct"]["macro_f1"] <= 1
    assert "baseline" in entry
    history = (results / "history.csv").read_text().splitlines()
    assert len(history) == 3 and "val_punct_macro_f1" in history[0]

    # A completed seed is skipped, never overwritten.
    train.main(["--config", str(cfg_path), "--all-seeds"])
    assert json.loads((results / "metrics.json").read_text()) == metrics


def test_dataset_batches_have_weights(tmp_path, configs_dir, tiny_assets):
    cfg_path = _config(tmp_path, configs_dir, tiny_assets)
    build_data.main(["--config", str(cfg_path), "--sets", "train"])
    arrays = ArraySet.load(tmp_path / "proc" / "train")
    ds = CleanupDataset(arrays, batch_size=4, punct_class_weights=[0.5, 2, 3, 4], seed=1)
    ids, y, sw = ds[0]
    assert ids.shape == (4, 32) and ids.dtype == np.int32
    ignored = np.asarray(arrays.punct[np.sort(ds._order[:4])]) == IGNORE
    assert (sw["punct"][ignored] == 0).all() and (sw["casing"][ignored] == 0).all()
    assert (y["punct"] >= 0).all()
    first = ds._order.copy()
    ds.on_epoch_end()
    assert not np.array_equal(first, ds._order)
    assert len(CleanupDataset(arrays, 4, max_batches=2)) == 2


def test_rule_baseline(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    words = label_text("well, i think i'm right. ice is cold.")
    enc = encode_windows(tok, [words], max_len=32)
    arrays = ArraySet(enc.token_ids, enc.casing, enc.punct, {})
    pred = rule_baseline(arrays, tok)
    pos = np.flatnonzero(enc.punct[0] != IGNORE)
    forms = [w.form for w in words]
    casing = dict(zip(forms, pred["casing"][0, pos], strict=False))
    assert casing["well"] == CAPITALISED and casing["i"] == CAPITALISED
    assert casing["i'm"] == CAPITALISED and casing["think"] == LOWER and casing["ice"] == LOWER
    assert pred["punct"][0, pos[-1]] == PERIOD and (pred["punct"][0, pos[:-1]] == 0).all()
    perfect = score(arrays, {"casing": np.where(enc.casing < 0, 0, enc.casing),
                             "punct": np.where(enc.punct < 0, 0, enc.punct)})  # fmt: skip
    assert perfect["punct"]["macro_f1"] == 1.0 and perfect["casing"]["accuracy"] == 1.0


def test_model2_config_has_plan_values(configs_dir):
    cfg = load_config(configs_dir / "model2.yaml")
    assert cfg["data"]["train_sources"] == {"wikitext_train": 1, "dialogsum_train": 5}
    assert cfg["data"]["window_words"] == 64 and cfg["data"]["max_len"] == 128
    assert cfg["eval"]["noise_levels"] == [0.0, 0.05, 0.10, 0.20]


def test_interrupted_run_resumes(tmp_path, configs_dir, tiny_assets):
    cfg_path = _config(tmp_path, configs_dir, tiny_assets)
    cfg_path.write_text(
        cfg_path.read_text()
        .replace("  epochs: 2\n", "  epochs: 3\n")
        .replace("max_steps: 6", "max_steps: 9")
        + "max_hours: 0.0001\n"  # 0.36 s: the guard stops after the first epoch
    )
    build_data.main(["--config", str(cfg_path)])
    train.main(["--config", str(cfg_path), "--seed", "0"])
    results = tmp_path / "results" / "m2_test" / "seed_0"
    assert json.loads((results / "manifest.json").read_text())["status"] == "interrupted"
    assert not (results / "metrics.json").exists()
    assert (tmp_path / "ckpt" / "seed_0" / "backup").exists()
    assert len((results / "history.csv").read_text().splitlines()) == 2  # header + epoch 0

    cfg_path.write_text(cfg_path.read_text().replace("max_hours: 0.0001", "max_hours: 1"))
    train.main(["--config", str(cfg_path), "--seed", "0"])
    manifest = json.loads((results / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["epochs_this_session"] == 2
    epochs = [line.split(",")[0] for line in (results / "history.csv").read_text().splitlines()]
    assert epochs == ["epoch", "0", "1", "2"]
    assert not (tmp_path / "ckpt" / "seed_0" / "backup").exists()
