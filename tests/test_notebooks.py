"""The Kaggle notebooks are self-contained, and what they save loads with the repo's code.

The notebooks do not import, clone or read anything from this repo. The only contract is the
saved files: custom layers, losses and optimizers are registered under ``dialogue_models`` with
the same class names and constructor arguments as in ``src/``, so ``keras.saving.load_model``
works here, and labels / predictions use the formats ``src/`` reads.

``RUN_NOTEBOOKS=1 uv run pytest tests/test_notebooks.py`` also executes each notebook in smoke
mode (CPU, ~1-2 min each, downloads the datasets once; Model 1 uses a fake teacher, never Qwen)
and loads its outputs with ``src/``.
"""

import inspect
import json
import os
import sys
from pathlib import Path

import pytest

from src.layers import embeddings, heads, transformer
from src.model1_scorer import losses, mlm_model
from src.training import LayerwiseAdamW

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "notebooks" / "kaggle"
NOTEBOOKS = sorted(KAGGLE.glob("*.ipynb"))
CORE = (
    embeddings.TokenAndPositionEmbedding,
    transformer.TransformerEncoderBlock,
    transformer.TransformerEncoder,
)
MODEL1 = (
    mlm_model.MLMHead,
    mlm_model.MLMPretrainer,
    heads.GatherPositions,
    losses.ImportanceLoss,
    LayerwiseAdamW,
)


def _code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _init_params(code: str, name: str) -> list[str]:
    """Constructor parameter names of class ``name`` in notebook code (``[]`` if no __init__)."""
    start = code.index(f"class {name}(")
    end = code.find("\nclass ", start + 1)
    body = code[start : end if end > 0 else len(code)]
    if "def __init__(" not in body:
        return []
    begin = body.index("def __init__(") + len("def __init__(")
    sig = body[begin : body.index("):", begin)]
    params = [a.split("=")[0].strip() for a in sig.split(",")]
    return [p for p in params if p and p not in ("self", "**kwargs")]


def test_one_notebook_per_model():
    assert [p.name for p in NOTEBOOKS] == ["model1_scorer.ipynb", "model2_cleanup.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_self_contained(path):
    code = _code(path)
    assert code.startswith('import os\n\nos.environ["KERAS_BACKEND"] = "jax"')
    for forbidden in (
        "from src",
        "import src",
        "git clone",
        "github.com",
        "%%writefile",
        "configs/",
        "import torch",
        "from torch",
        "tensorflow",
    ):
        assert forbidden not in code, f"{path.name} must not use {forbidden!r}"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), (
        "commit notebooks without outputs"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_classes_match_src(path):
    """Same registered class names and constructor arguments as ``src/``."""
    code = _code(path)
    expected = CORE + (MODEL1 if path.stem == "model1_scorer" else ())
    for cls in expected:
        assert f"class {cls.__name__}(" in code, cls.__name__
        params = inspect.signature(cls.__init__).parameters if "__init__" in vars(cls) else {}
        args = [p for p in params if p not in ("self", "kwargs")]
        assert _init_params(code, cls.__name__) == args, cls.__name__
    n = code.count('register_keras_serializable(package="dialogue_models")')
    assert n == len(expected)


def test_model1_notebook_teacher_runs_outside_jax():
    """Teachers run as a subprocess (own venv) before the notebook imports JAX."""
    code = _code(KAGGLE / "model1_scorer.ipynb")
    assert (
        code.index("TEACHER_SCRIPT = r'''")
        < code.index("subprocess.Popen")
        < code.index("import jax")
    )
    script = code[
        code.index("TEACHER_SCRIPT = r'''") : code.index("'''\n", code.index("TEACHER_SCRIPT"))
    ]
    assert "import jax" not in script and "import keras" not in script


def _execute(path: Path, tmp_path: Path) -> Path:
    import nbformat
    from nbclient import NotebookClient

    spec = tmp_path / "jupyter" / "kernels" / "project"
    spec.mkdir(parents=True)
    argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    (spec / "kernel.json").write_text(
        json.dumps({"argv": argv, "language": "python", "display_name": "project"})
    )
    work = tmp_path / "working"
    work.mkdir()
    env = {
        "JUPYTER_PATH": str(tmp_path / "jupyter"),
        "SMOKE": "1",
        "WORK_DIR": str(work),
        "INPUT_DIR": str(tmp_path / "input"),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        nb = nbformat.read(path, as_version=4)
        NotebookClient(
            nb, timeout=1200, kernel_name="project", resources={"metadata": {"path": str(work)}}
        ).execute()
    finally:
        for k, v in old.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    return work


RUN = pytest.mark.skipif(os.environ.get("RUN_NOTEBOOKS") != "1", reason="set RUN_NOTEBOOKS=1")


@RUN
def test_model2_notebook_smoke_run_and_load_here(tmp_path):
    import keras
    import numpy as np

    import src.layers.transformer  # noqa: F401  (registers the layers)
    from src.tokenizer.tokenizer import CLS_ID, SEP_ID, load_tokenizer

    work = _execute(KAGGLE / "model2_cleanup.ipynb", tmp_path)
    ckpt = work / "checkpoints" / "model2"
    model = keras.saving.load_model(ckpt / "seed_0" / "best.keras")
    tok = load_tokenizer(ckpt / "tokenizer")
    ids = [CLS_ID, *tok.encode("hello how are you", add_special_tokens=False).ids, SEP_ID]
    out = model.predict(np.array([ids], dtype=np.int32), verbose=0)
    assert out["casing"].shape == (1, len(ids), 3) and out["punct"].shape == (1, len(ids), 4)
    assert json.loads((work / "manifest_model2_cleanup.json").read_text())["complete_seeds"] == [0]


@RUN
def test_model1_notebook_smoke_run_and_load_here(tmp_path):
    import keras
    import numpy as np

    import src.layers.heads  # noqa: F401  (registers the classes the models need)
    import src.model1_scorer.losses  # noqa: F401
    import src.training  # noqa: F401
    from src.model1_scorer.data import StudentSet, load_targets
    from src.model1_scorer.model import build_student, load_encoder_weights
    from src.model1_scorer.train import model_inputs
    from src.tokenizer.tokenizer import load_tokenizer

    work = _execute(KAGGLE / "model1_scorer.ipynb", tmp_path)
    manifest = json.loads((work / "manifest_model1_scorer.json").read_text())
    assert manifest["finished_all"] and manifest["complete_seeds"]["model1_student_A"] == [0]
    ckpt = work / "checkpoints"
    load_tokenizer(ckpt / "model1_tokenizer")

    # The notebook's students and MLM encoder load and run with src/.
    test = StudentSet.load(work / "data" / "processed" / "model1" / "test")
    student = keras.saving.load_model(ckpt / "model1_student_A" / "seed_0" / "best.keras")
    out = student.predict(model_inputs(test, np.arange(2)), verbose=0)
    assert out["importance"].shape[0] == 2 and out["tags"].shape[-1] == 4
    model_cfg = manifest["model"]
    fresh = build_student(model_cfg, use_tags=True)
    load_encoder_weights(fresh, ckpt / "mlm" / "encoder.keras")

    # Its labels and predictions are in the formats src/ reads.
    for source in ("qwen7b", "qwen1_5b", "oracle"):
        targets = load_targets(work / "data" / "labels" / source, "test", test, source)
        assert targets.labelled.all()
    preds = work / "results" / "model1_student_A" / "seed_0" / "test_predictions.jsonl"
    first = json.loads(preds.read_text().splitlines()[0])
    assert set(first) == {"id", "scores", "tags"}
    assert (work / "results" / "tables" / "model1_preview.md").exists()
