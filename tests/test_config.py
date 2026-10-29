"""Tests for config loading and for config values matching PLAN.md."""

from pathlib import Path

import pytest

from src.config import ConfigError, deep_merge, load_config

CONFIG_FILES = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))


def _encoder_params(m: dict) -> int:
    """Approximate parameter count of an encoder described by a ``model`` config section."""
    d, ffn, n = m["d_model"], m["ffn_dim"], m["num_layers"]
    emb = m["vocab_size"] * d + m["max_len"] * d + m.get("num_speakers", 0) * d
    attn = 4 * d * d + 4 * d
    ffn_p = 2 * d * ffn + ffn + d
    norms = 4 * d
    return emb + n * (attn + ffn_p + norms) + 2 * d


def test_deep_merge_nested_override():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    merged = deep_merge(base, {"b": {"d": 4}, "e": 5})
    assert merged == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}
    assert base == {"a": 1, "b": {"c": 2, "d": 3}}  # input not mutated


def test_base_inheritance(tmp_path):
    (tmp_path / "parent.yaml").write_text("name: p\nseed: 1\nmodel: {d: 8, l: 2}\n")
    (tmp_path / "child.yaml").write_text("base: parent.yaml\nname: c\nmodel: {l: 1}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"name": "c", "seed": 1, "model": {"d": 8, "l": 1}}


def test_cyclic_inheritance_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\nname: a\nseed: 0\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nname: b\nseed: 0\n")
    with pytest.raises(ConfigError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize(
    "body",
    ["name: x\n", "name: x\nseed: '1'\n", "name: x\nseeds: []\n", "seed: 0\n", "- 1\n"],
)
def test_invalid_configs_rejected(tmp_path, body):
    (tmp_path / "bad.yaml").write_text(body)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "bad.yaml")


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_every_config_loads(path):
    cfg = load_config(path)
    assert "base" not in cfg
    assert cfg["description"]


def test_config_names_unique():
    names = [load_config(p)["name"] for p in CONFIG_FILES]
    assert len(names) == len(set(names))


def test_tokenizer_special_tokens(configs_dir):
    cfg = load_config(configs_dir / "tokenizer.yaml")
    specials = cfg["special_tokens"]
    assert specials[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    assert specials[5:] == [f"[S{i}]" for i in range(1, 9)]
    assert cfg["vocab_size"] == 16000


def test_vocab_size_shared(configs_dir):
    vocab = load_config(configs_dir / "tokenizer.yaml")["vocab_size"]
    for p in CONFIG_FILES:
        cfg = load_config(p)
        if "model" in cfg:
            assert cfg["model"]["vocab_size"] == vocab, p.name


def test_model2_matches_plan(configs_dir):
    cfg = load_config(configs_dir / "model2.yaml")
    m = cfg["model"]
    assert (m["d_model"], m["num_layers"], m["num_heads"], m["ffn_dim"], m["max_len"]) == (
        256,
        4,
        4,
        1024,
        128,
    )
    assert 6.5e6 < _encoder_params(m) < 8e6
    assert cfg["data"]["noise"] == {
        "substitution": 0.05,
        "deletion": 0.03,
        "insertion": 0.02,
        "substitution_vocab_size": 20000,
    }
    assert cfg["optimizer"]["peak_lr"] == 5e-4
    assert cfg["training"]["batch_size"] == 128


def test_model1_encoder_matches_the_pretrained_mlm_encoder(configs_dir):
    """The v3 sentence encoder loads the MLM encoder pretrained on Kaggle; its architecture is
    recorded in that run's manifest (results/kaggle/manifest_model1_scorer.json)."""
    import json

    manifest = configs_dir.parent / "results" / "kaggle" / "manifest_model1_scorer.json"
    mlm = json.loads(manifest.read_text())["model"]
    v3 = load_config(configs_dir / "model1_v3.yaml")["model"]
    keys = [
        "vocab_size",
        "d_model",
        "num_layers",
        "num_heads",
        "ffn_dim",
        "max_len",
        "num_speakers",
    ]
    assert [v3[k] for k in keys] == [mlm[k] for k in keys] == [16000, 384, 6, 6, 1536, 512, 8]
    assert 15e6 < _encoder_params(v3) < 19e6


MODEL_TINY = sorted(p.name for p in CONFIG_FILES if "_tiny" in p.name and "tokenizer" not in p.name)


@pytest.mark.parametrize("tiny", MODEL_TINY)
def test_tiny_configs_are_small(configs_dir, tiny):
    cfg = load_config(configs_dir / tiny)
    assert cfg["training"]["max_steps"] <= 50
    assert cfg["training"]["mixed_precision"] == "float32"
    assert _encoder_params(cfg["model"]) < 1e6
    assert "tiny" in cfg["paths"]["output_dir"]
