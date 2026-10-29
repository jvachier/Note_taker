"""Static checks for the import rules in CLAUDE.md (scans ``src/`` and ``app/`` with ``ast``)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
APP = ROOT / "app"
TEACHER_DIRS = (SRC / "model1_scorer" / "teacher", SRC / "model1_notes" / "teacher")
EVAL_DIR = SRC / "eval"

BANNED_EVERYWHERE = {"torch", "tensorflow", "tensorflow_text", "tf_keras"}
EVAL_ONLY = {"bert_score", "rouge_score"}
TEACHER_ONLY = {"vllm", "transformers", "bitsandbytes", "awq"}
# Oracle labels (PLAN 5.4) are computed with ROUGE, outside the training code.
ROUGE_ALLOWED = {SRC / "model1_scorer" / "oracle.py"}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


SRC_FILES = sorted(SRC.rglob("*.py"))
APP_FILES = sorted(APP.rglob("*.py"))


@pytest.mark.parametrize("path", SRC_FILES + APP_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_import_rules(path):
    imports = _top_level_imports(path)
    assert not imports & BANNED_EVERYWHERE, f"banned framework import in {path}"
    if not any(path.is_relative_to(d) for d in TEACHER_DIRS):
        assert not imports & TEACHER_ONLY, f"teacher-only import outside teacher/: {path}"
    if not path.is_relative_to(EVAL_DIR):
        allowed = {"rouge_score"} if path in ROUGE_ALLOWED else set()
        assert not (imports & EVAL_ONLY) - allowed, f"eval-only import in {path}"


def test_scanner_detects_imports(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import torch.nn\nfrom tensorflow import data\nfrom . import x\n")
    assert _top_level_imports(f) == {"torch", "tensorflow"}


TEACHER_FILES = sorted(f for d in TEACHER_DIRS for f in d.rglob("*.py"))


@pytest.mark.parametrize("path", TEACHER_FILES, ids=lambda p: p.name)
def test_teacher_code_never_imports_jax_or_keras(path):
    """The teachers run in their own vLLM environment, never with training (CLAUDE.md)."""
    assert not _top_level_imports(path) & {"jax", "keras", "jaxlib"}, path


@pytest.mark.parametrize(
    "module",
    [
        "src.model2_cleanup.train",
        "src.model1_scorer.mlm_train",
        "src.model1_scorer.train",
        "src.model1_notes.train",
    ],
)
def test_training_code_does_not_load_eval_metrics(module):
    """ROUGE / BERTScore are evaluation-only, also indirectly (CLAUDE.md)."""
    import subprocess
    import sys

    code = (
        f"import sys, {module}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'rouge_score', 'bert_score', 'torch'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_teacher_package_runs_without_ml_frameworks():
    """The Qwen teacher code must import in a venv without JAX / Keras (CLAUDE.md)."""
    import subprocess
    import sys

    code = (
        "import sys, src.model1_notes.teacher.label, src.model1_notes.teacher.gate; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'jax', 'keras', 'tokenizers', 'torch'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_student_training_never_loads_the_teachers():
    """Teachers and baselines run only in Notebook 1; training reads their cached outputs."""
    import subprocess
    import sys

    code = (
        "import sys, src.model1_notes.train; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'keras_hub', 'vllm'} "
        "or m.startswith(('src.model1_notes.encoder_teachers', 'src.model1_notes.teacher.label'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
