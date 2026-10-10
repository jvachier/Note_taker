"""Static checks for the import rules in CLAUDE.md (scans ``src/`` with ``ast``)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
TEACHER_DIR = SRC / "model1_scorer" / "teacher"
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


@pytest.mark.parametrize("path", SRC_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_import_rules(path):
    imports = _top_level_imports(path)
    assert not imports & BANNED_EVERYWHERE, f"banned framework import in {path}"
    if not path.is_relative_to(TEACHER_DIR):
        assert not imports & TEACHER_ONLY, f"teacher-only import outside teacher/: {path}"
    if not path.is_relative_to(EVAL_DIR):
        allowed = {"rouge_score"} if path in ROUGE_ALLOWED else set()
        assert not (imports & EVAL_ONLY) - allowed, f"eval-only import in {path}"


def test_scanner_detects_imports(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import torch.nn\nfrom tensorflow import data\nfrom . import x\n")
    assert _top_level_imports(f) == {"torch", "tensorflow"}
