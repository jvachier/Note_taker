"""Kaggle outputs are copied back into the repo layout, without resume-only state."""

import importlib.util
import json
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle"


def _fetch():
    spec = importlib.util.spec_from_file_location("fetch_outputs", KAGGLE / "fetch_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_output(root: Path) -> Path:
    out = root / "download" / "nested"
    seed = out / "checkpoints" / "model2" / "seed_0"
    (seed / "backup").mkdir(parents=True)
    (seed / "best.keras").write_bytes(b"model")
    (seed / "backup" / "state.json").write_text("{}")
    (out / "checkpoints" / "model2" / "tokenizer").mkdir()
    (out / "checkpoints" / "model2" / "tokenizer" / "tokenizer.json").write_text("{}")
    res = out / "results" / "model2_cleanup" / "seed_0"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text("{}")
    (out / "data" / "processed").mkdir(parents=True)
    (out / "manifest_model2_cleanup.json").write_text(json.dumps({"complete_seeds": [0]}))
    return root / "download"


def test_install_copies_models_and_results_only(tmp_path):
    fetch = _fetch()
    src = fetch.output_root(_fake_output(tmp_path), "model2_cleanup")
    repo = tmp_path / "repo"
    fetch.install(src, "model2_cleanup", repo)
    ckpt = repo / "checkpoints" / "model2"
    assert (ckpt / "seed_0" / "best.keras").read_bytes() == b"model"
    assert (ckpt / "tokenizer" / "tokenizer.json").exists()
    assert not (ckpt / "seed_0" / "backup").exists()
    assert not (repo / "data").exists()
    assert (repo / "results" / "model2_cleanup" / "seed_0" / "metrics.json").exists()
    assert (repo / "results" / "kaggle" / "manifest_model2_cleanup.json").exists()


def test_output_without_manifest_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="manifest_model2_cleanup.json"):
        _fetch().output_root(tmp_path, "model2_cleanup")


def test_smoke_output_is_rejected(tmp_path):
    out = _fake_output(tmp_path)
    (out / "nested" / "manifest_model2_cleanup.json").write_text(json.dumps({"smoke": True}))
    with pytest.raises(SystemExit, match="smoke"):
        _fetch().main(["model2_cleanup", "--from-dir", str(out)])


def test_default_kernel_slug(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    assert _fetch().default_kernel("model2_cleanup") == "someone/model2-cleanup"
