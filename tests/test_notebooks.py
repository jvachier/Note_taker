"""The Kaggle notebooks are self-contained, and what they save loads with the repo's code.

The notebooks do not import, clone or read anything from this repo. The only contract is the
saved model: its custom layers are registered under ``dialogue_models`` with the same class
names and configs as ``src/layers``, so ``keras.saving.load_model`` works here.

``RUN_NOTEBOOKS=1 uv run pytest tests/test_notebooks.py`` also executes each notebook in smoke
mode (CPU, ~1-2 min, downloads the datasets once) and loads the saved model with ``src/``.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "notebooks" / "kaggle"
NOTEBOOKS = sorted(KAGGLE.glob("*.ipynb"))


def _code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_one_notebook_per_model():
    assert [p.name for p in NOTEBOOKS] == ["model2_cleanup.ipynb"]


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
        "torch",
        "tensorflow",
    ):
        assert forbidden not in code, f"{path.name} must not use {forbidden!r}"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), (
        "commit notebooks without outputs"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_layers_match_src(path):
    """Same registered layer names and constructor arguments as ``src/layers``."""
    import inspect

    from src.layers import embeddings, transformer

    code = _code(path)
    for cls in (embeddings.TokenAndPositionEmbedding, transformer.TransformerEncoderBlock,
                transformer.TransformerEncoder):  # fmt: skip
        assert f"class {cls.__name__}(keras.layers.Layer)" in code
        sig = inspect.signature(cls.__init__).parameters
        args = [p for p in sig if p not in ("self", "kwargs")]
        start = code.index("def __init__(", code.index(f"class {cls.__name__}("))
        signature = code[start + len("def __init__(") : code.index("):", start)]
        params = [a.split("=")[0].strip() for a in signature.split(",")]
        assert [p for p in params if p and p not in ("self", "**kwargs")] == args
    assert code.count('register_keras_serializable(package="dialogue_models")') >= 3


@pytest.mark.skipif(os.environ.get("RUN_NOTEBOOKS") != "1", reason="set RUN_NOTEBOOKS=1")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_smoke_run_and_load_here(path, tmp_path):
    import keras
    import nbformat
    import numpy as np
    from nbclient import NotebookClient

    import src.layers.transformer  # noqa: F401  (registers the layers)
    from src.tokenizer.tokenizer import CLS_ID, SEP_ID, load_tokenizer

    spec = tmp_path / "jupyter" / "kernels" / "project"
    spec.mkdir(parents=True)
    argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    (spec / "kernel.json").write_text(json.dumps({"argv": argv, "language": "python",
                                                  "display_name": "project"}))  # fmt: skip
    work = tmp_path / "working"
    work.mkdir()
    env = {"JUPYTER_PATH": str(tmp_path / "jupyter"), "SMOKE": "1", "WORK_DIR": str(work),
           "INPUT_DIR": str(tmp_path / "input")}  # fmt: skip
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=900, kernel_name="project",
                       resources={"metadata": {"path": str(work)}}).execute()  # fmt: skip
    finally:
        for k, v in old.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)

    ckpt = work / "checkpoints" / "model2"
    model = keras.saving.load_model(ckpt / "seed_0" / "best.keras")
    tok = load_tokenizer(ckpt / "tokenizer")
    ids = [CLS_ID, *tok.encode("hello how are you", add_special_tokens=False).ids, SEP_ID]
    out = model.predict(np.array([ids], dtype=np.int32), verbose=0)
    assert out["casing"].shape == (1, len(ids), 3) and out["punct"].shape == (1, len(ids), 4)
    assert json.loads((work / "manifest_model2_cleanup.json").read_text())["complete_seeds"] == [0]
