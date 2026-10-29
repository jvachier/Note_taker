"""Bring a Kaggle notebook's trained models and results into this repo.

Usage::

    uv run python notebooks/kaggle/fetch_outputs.py model2_cleanup
    uv run python notebooks/kaggle/fetch_outputs.py model2_cleanup --kernel <user>/<slug>
    uv run python notebooks/kaggle/fetch_outputs.py model2_cleanup --from-dir ~/Downloads/out

The notebooks are self-contained and know nothing about this repo; this script is the only
link. It downloads the latest output of the notebook with the Kaggle CLI
(``kaggle kernels output``; credentials in ``~/.kaggle/kaggle.json``) into
``data/kaggle/<notebook>/``, or uses an already unzipped download (``--from-dir``), and copies
the parts listed in ``PULL`` to the same relative paths here:

    model2_cleanup  checkpoints/model2/ (best.keras per seed, tokenizer), results/
    model1_scorer   checkpoints/{mlm,model1_student_*,model1_tokenizer}/, results/,
                    data/labels/{qwen14b,qwen1_5b,oracle}/
    model1_teachers data/labels/model1_v3/ (Qwen, RoBERTa, DistilBERT tables, gate, raw thoughts)
    model1_notes    checkpoints/model1_v3_*/, results/

Resume-only state (``backup/``) is skipped. The notebook manifest goes to ``results/kaggle/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# What each notebook brings back: relative paths inside its output (= inside the repo).
PULL: dict[str, list[str]] = {
    "model2_cleanup": ["checkpoints/model2", "results"],
    # MLM encoder, students, tokenizer; results; teacher + oracle labels (for the evaluation).
    "model1_scorer": ["checkpoints", "results", "data/labels"],
    # Model 1 v3: teacher outputs (Notebook 1), students and their results (Notebook 2).
    "model1_teachers": ["data/labels/model1_v3"],
    "model1_notes": ["checkpoints", "results"],
}
SKIP = shutil.ignore_patterns("backup", "__pycache__", ".ipynb_checkpoints")


def default_kernel(notebook: str) -> str:
    """``<user>/<slug>``: Kaggle derives the slug from the notebook title."""
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        creds = Path.home() / ".kaggle" / "kaggle.json"
        user = json.loads(creds.read_text())["username"]
    return f"{user}/{notebook.replace('_', '-').lower()}"


def download(kernel: str, dest: Path) -> None:
    """Download the latest output of ``kernel`` into ``dest`` (replacing its contents)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subprocess.run(["kaggle", "kernels", "output", kernel, "-p", str(dest)], check=True)


def output_root(src: Path, notebook: str) -> Path:
    """The directory holding ``manifest_<notebook>.json`` (downloads may add a level)."""
    name = f"manifest_{notebook}.json"
    found = sorted(src.rglob(name), key=lambda p: len(p.parts))
    if not found:
        raise SystemExit(f"{name} not found under {src}: not a finished {notebook} output.")
    return found[0].parent


def install(src: Path, notebook: str, repo: Path) -> list[Path]:
    """Copy the parts of ``notebook``'s output listed in ``PULL`` into ``repo``."""
    copied = []
    for rel in PULL[notebook]:
        if not (src / rel).exists():
            print(f"  (not in output: {rel})")
            continue
        shutil.copytree(src / rel, repo / rel, dirs_exist_ok=True, ignore=SKIP)
        copied.append(repo / rel)
    manifests = repo / "results" / "kaggle"
    manifests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / f"manifest_{notebook}.json", manifests)
    copied.append(manifests / f"manifest_{notebook}.json")
    return copied


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", choices=sorted(PULL))
    parser.add_argument("--kernel", help="<user>/<slug> (default: from kaggle.json + name)")
    parser.add_argument("--from-dir", type=Path, help="use an already downloaded output")
    args = parser.parse_args(argv)

    if args.from_dir:
        src = args.from_dir.expanduser().resolve()
    else:
        src = ROOT / "data" / "kaggle" / args.notebook
        kernel = args.kernel or default_kernel(args.notebook)
        print(f"downloading {kernel} -> {src}")
        download(kernel, src)
    src = output_root(src, args.notebook)

    manifest = json.loads((src / f"manifest_{args.notebook}.json").read_text())
    if manifest.get("smoke"):
        raise SystemExit("This output is a smoke run (SMOKE=1), not a real one.")
    print(f"output finished {manifest.get('finished')}, seeds {manifest.get('complete_seeds')}")
    for path in install(src, args.notebook, ROOT):
        print("  ->", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
