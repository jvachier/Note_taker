"""Release packaging and the demo app: dialogue parsing, model loading, Dash callbacks."""

import json

import pytest
import yaml

from app.text import changed_words, parse_dialogue
from src.model1_scorer.model import build_student
from src.model2_cleanup.model import build_cleanup_model
from src.publish.package import main as package_main

SMALL = {
    "vocab_size": 400,
    "max_len": 32,
    "d_model": 16,
    "num_layers": 1,
    "num_heads": 2,
    "ffn_dim": 32,
    "dropout": 0.0,
    "init_std": 0.02,
}


def test_parse_dialogue_assigns_speakers_in_order():
    d = parse_dialogue("Anna: Hi.\n\nBen: Hello!\n#Person1#: Bye.\nAnna: See you.\nand later")
    assert d.speakers == [1, 2, 3, 1, 1]
    assert d.names == ["Anna", "Ben", "Person1"]
    assert d.utterances[-1] == "and later"
    assert d.name(1) == "Ben"


def test_parse_dialogue_errors():
    with pytest.raises(ValueError, match="at least one line"):
        parse_dialogue("  \n ")
    with pytest.raises(ValueError, match="At most 8 speakers"):
        parse_dialogue("\n".join(f"S{i}: hi" for i in range(9)))
    assert parse_dialogue("no speaker here").speakers == [1]


def test_changed_words_marks_restored_words():
    assert changed_words("hi john, ok", "Hi john OK.") == [
        ("Hi", True),
        ("john", False),
        ("OK.", True),
    ]
    assert changed_words("a b", "A") == [("A", False)]  # misaligned: no highlighting


@pytest.fixture
def release(tmp_path, tiny_assets):
    """Tiny checkpoints + results for both models, packaged by :mod:`src.publish.package`."""
    runs = {
        "cleanup": ("model2_tiny", build_cleanup_model(SMALL), [["validation", "f1"]]),
        "scorer": (
            "student_tiny",
            build_student({**SMALL, "num_speakers": 8}, use_tags=True),
            [["validation", "spearman"]],
        ),
    }
    models = {}
    for kind, (name, model, (metric,)) in runs.items():
        for seed, value in ((0, 0.2), (1, 0.4)):
            run_dir = tmp_path / "results" / name / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(json.dumps({metric[0]: {metric[1]: value}}))
            ckpt = tmp_path / "checkpoints" / name / f"seed_{seed}"
            ckpt.mkdir(parents=True)
            model.compile(optimizer="adam")  # saved with optimizer state, like training
            model.save(ckpt / "best.keras")
        run_cfg = {
            "name": name,
            "seeds": [0, 1],
            "paths": {
                "results_dir": str(tmp_path / "results"),
                "output_dir": str(tmp_path / "checkpoints" / name),
            },
            "model": SMALL,
            "data": {"window_words": 8, "max_len": 32},
            "inference": {"top_k": 2},
        }
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(run_cfg))
        models[f"repo-{kind}"] = {
            "kind": kind,
            "config": str(tmp_path / f"{name}.yaml"),
            "metric": metric,
        }
    cfg = {
        "name": "release",
        "seed": 0,
        "output_dir": str(tmp_path / "release"),
        "tokenizer_dir": str(tiny_assets["tokenizer"]),
        "models": models,
    }
    (tmp_path / "release.yaml").write_text(yaml.safe_dump(cfg))
    package_main(["--config", str(tmp_path / "release.yaml")])
    return tmp_path / "release"


def test_package_writes_hub_layout(release):
    for kind in ("cleanup", "scorer"):
        folder = release / f"repo-{kind}"
        assert {p.name for p in folder.iterdir()} == {
            "model.keras",
            "tokenizer.json",
            "config.json",
        }
        config = json.loads((folder / "config.json").read_text())
        assert config["kind"] == kind
        assert config["source"]["seed"] == 1  # best validation value
    assert json.loads((release / "repo-scorer" / "config.json").read_text())["inference"] == {
        "max_len": 32,
        "top_k": 2,
    }


def test_models_and_dash_callbacks(release, monkeypatch):
    pytest.importorskip("dash")
    from app.main import create_app
    from app.models import Bundle, Models

    with pytest.raises(ValueError, match="expected scorer"):
        Bundle.load(str(release / "repo-cleanup"), "scorer")
    monkeypatch.setenv("MODEL2_SOURCE", str(release / "repo-cleanup"))
    monkeypatch.setenv("MODEL1_SOURCE", str(release / "repo-scorer"))
    models = Models.from_env()
    cleaned, _ = models.clean("hello there how are you")
    assert len(cleaned.split()) == 5
    scored, _ = models.score([1, 2, 1], ["Hi.", "Can you send it?", "Yes."])
    assert len(scored.scores) == 3 and len(scored.selected) == 2

    client = create_app(models).server.test_client()
    assert client.get("/").status_code == 200

    def call(output, inputs, state):
        body = {
            "output": f"{output}.children",
            "outputs": {"id": output, "property": "children"},
            "inputs": inputs,
            "state": state,
            "changedPropIds": [f"{inputs[0]['id']}.{inputs[0]['property']}"],
        }
        response = client.post("/_dash-update-component", json=body)
        assert response.status_code == 200
        return json.dumps(response.get_json())

    out = call(
        "clean-output",
        [{"id": "clean-btn", "property": "n_clicks", "value": 1}],
        [{"id": "clean-input", "property": "value", "value": "hello there how are you"}],
    )
    assert "Result" in out
    out = call(
        "notes-output",
        [{"id": "notes-btn", "property": "n_clicks", "value": 1}],
        [
            {"id": "notes-input", "property": "value", "value": "a: hi\nb: send it\na: ok"},
            {"id": "notes-clean", "property": "value", "value": ["clean"]},
        ],
    )
    assert "Notes" in out and "Importance of every utterance" in out
    out = call(
        "notes-output",
        [{"id": "notes-btn", "property": "n_clicks", "value": 1}],
        [
            {"id": "notes-input", "property": "value", "value": ""},
            {"id": "notes-clean", "property": "value", "value": []},
        ],
    )
    assert "at least one line" in out
