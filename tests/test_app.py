"""Release packaging and the demo app: dialogue parsing, model loading, Dash callbacks."""

import json

import pytest
import yaml

from app.text import (
    MAX_CHARS,
    changed_words,
    meeting_sentences,
    parse_dialogue,
    split_sentences,
)
from src.model1_notes.metrics import ZONES
from src.model1_notes.model import build_notes_model
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
    with pytest.raises(ValueError, match="accepts up to"):
        parse_dialogue("a: " + "x" * MAX_CHARS)


def test_changed_words_marks_restored_words():
    assert changed_words("hi john, ok", "Hi john OK.") == [
        ("Hi", True),
        ("john", False),
        ("OK.", True),
    ]
    assert changed_words("a b", "A") == [("A", False)]  # misaligned: no highlighting


NOTES = {
    **SMALL,
    "max_len": 16,
    "meeting_layers": 1,
    "num_speakers": 8,
    "num_zones": 5,
    "num_reasons": 9,
    "num_span_tags": 5,
}


def test_split_sentences_and_meeting_sentences():
    assert split_sentences("Okay. We ship on Monday! Who tests it?") == [
        "Okay.",
        "We ship on Monday!",
        "Who tests it?",
    ]
    assert split_sentences("no punctuation here") == ["no punctuation here"]
    assert split_sentences("Ask him about the rent. increase, too.") == [
        "Ask him about the rent. increase, too."
    ]
    d = parse_dialogue("Anna: Hi. Shall we start?\nBen: Yes.")
    assert meeting_sentences(d) == (["Anna", "Anna", "Ben"], ["Hi.", "Shall we start?", "Yes."])


@pytest.fixture
def release(tmp_path, tiny_assets):
    """Tiny checkpoints + results for both models, packaged by :mod:`src.publish.package`."""
    runs = {
        "cleanup": ("model2_tiny", build_cleanup_model(SMALL), [["validation", "f1"]]),
        "notes": (
            "notes_tiny",
            build_notes_model(NOTES, max_sentences=16, sentence_tokens=16),
            [["validation", "ami", "importance_f1"]],
        ),
    }
    models = {}
    for kind, (name, model, (metric,)) in runs.items():
        for seed, value in ((0, 0.2), (1, 0.4)):
            run_dir = tmp_path / "results" / name / f"seed_{seed}"
            run_dir.mkdir(parents=True)
            metrics = {"threshold": 0.3, "zone_thresholds": {"decision": 0.2, "action_item": 0.25}}
            leaf = {metric[-1]: value}
            for key in reversed(metric[:-1]):
                leaf = {key: leaf}
            (run_dir / "metrics.json").write_text(json.dumps({**leaf, **metrics}))
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
            "model": NOTES if kind == "notes" else SMALL,
            "data": {
                "window_words": 8,
                "max_len": 32,
                "max_sentence_tokens": 16,
                "window_overlap": 4,
            },
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
    for kind in ("cleanup", "notes"):
        folder = release / f"repo-{kind}"
        assert {p.name for p in folder.iterdir()} == {
            "model.keras",
            "tokenizer.json",
            "config.json",
        }
        config = json.loads((folder / "config.json").read_text())
        assert config["kind"] == kind
        assert config["source"]["seed"] == 1  # best validation value
    assert json.loads((release / "repo-notes" / "config.json").read_text())["inference"] == {
        "max_sentence_tokens": 16,
        "window_overlap": 4,
        "threshold": 0.3,
        "zone_thresholds": {"decision": 0.2, "action_item": 0.25},
        "postprocess": None,
    }


def test_models_and_dash_callbacks(release, monkeypatch):
    pytest.importorskip("dash")
    from app.main import create_app
    from app.models import Bundle, Models

    with pytest.raises(ValueError, match="expected notes"):
        Bundle.load(str(release / "repo-cleanup"), "notes")
    monkeypatch.setenv("MODEL2_SOURCE", str(release / "repo-cleanup"))
    monkeypatch.setenv("MODEL1_SOURCE", str(release / "repo-notes"))
    models = Models.from_env()
    cleaned, _ = models.clean("hello there how are you")
    assert len(cleaned.split()) == 5
    speakers = ["Anna", "Ben"] * 12  # 24 sentences: two overlapping windows of 16
    notes, _ = models.annotate(speakers, [f"Sentence number {i}." for i in range(24)])
    assert [n.index for n in notes] == list(range(24))
    assert all(0 <= n.score <= 1 for n in notes) and {n.zone for n in notes} <= set(ZONES)

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
    assert "Notes" in out and "Importance of every sentence" in out and "Action items" in out
    out = call(
        "notes-output",
        [{"id": "notes-btn", "property": "n_clicks", "value": 1}],
        [
            {"id": "notes-input", "property": "value", "value": ""},
            {"id": "notes-clean", "property": "value", "value": []},
        ],
    )
    assert "at least one line" in out


class FailingBackend:
    """A backend whose models always fail (the app must answer with a message, not crash)."""

    def clean(self, text):
        raise RuntimeError("model crashed")

    def annotate(self, speakers, sentences):
        raise RuntimeError("model crashed")


def test_callbacks_report_errors_and_too_long_input():
    pytest.importorskip("dash")
    from app.main import MODEL_ERROR, create_app

    client = create_app(FailingBackend()).server.test_client()

    def call(output, button, state):
        body = {
            "output": f"{output}.children",
            "outputs": {"id": output, "property": "children"},
            "inputs": [{"id": button, "property": "n_clicks", "value": 1}],
            "state": state,
            "changedPropIds": [f"{button}.n_clicks"],
        }
        response = client.post("/_dash-update-component", json=body)
        assert response.status_code == 200
        return json.dumps(response.get_json())

    clean = [{"id": "clean-input", "property": "value", "value": "hello there"}]
    assert MODEL_ERROR in call("clean-output", "clean-btn", clean)
    too_long = [{"id": "clean-input", "property": "value", "value": "x" * (MAX_CHARS + 1)}]
    assert "accepts up to" in call("clean-output", "clean-btn", too_long)
    notes = [
        {"id": "notes-input", "property": "value", "value": "a: hi\nb: ok"},
        {"id": "notes-clean", "property": "value", "value": ["clean"]},
    ]
    assert MODEL_ERROR in call("notes-output", "notes-btn", notes)


def test_source_from_env_uses_hf_namespace_never_a_hardcoded_account(monkeypatch, tmp_path):
    import app.models as m

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)  # no release/ folder here
    monkeypatch.delenv("MODEL1_SOURCE", raising=False)
    monkeypatch.delenv("HF_NAMESPACE", raising=False)
    with pytest.raises(ValueError, match="HF_NAMESPACE"):
        m.source_from_env("MODEL1_SOURCE")
    monkeypatch.setenv("HF_NAMESPACE", "someone")
    assert m.source_from_env("MODEL1_SOURCE") == "someone/meeting-notes-en-21m"
    (tmp_path / "release" / "meeting-notes-en-21m").mkdir(parents=True)
    assert m.source_from_env("MODEL1_SOURCE") == str(tmp_path / "release/meeting-notes-en-21m")
    monkeypatch.setenv("MODEL1_SOURCE", "/models/x")
    assert m.source_from_env("MODEL1_SOURCE") == "/models/x"


def test_fill_namespace_replaces_the_placeholder_in_hub_files(tmp_path):
    from src.publish.package import fill_namespace

    (tmp_path / "README.md").write_text('snapshot_download("<hf-user>/meeting-notes-en-21m")')
    (tmp_path / "example.py").write_text('REPO_ID = "<hf-user>/m"')
    (tmp_path / "config.json").write_text('{"x": "<hf-user>"}')  # data files are left alone
    fill_namespace(tmp_path, None)  # no namespace: placeholder kept (and a warning)
    assert "<hf-user>" in (tmp_path / "README.md").read_text()
    fill_namespace(tmp_path, "someone")
    assert (tmp_path / "README.md").read_text() == (
        'snapshot_download("someone/meeting-notes-en-21m")'
    )
    assert (tmp_path / "example.py").read_text() == 'REPO_ID = "someone/m"'
    assert "<hf-user>" in (tmp_path / "config.json").read_text()
