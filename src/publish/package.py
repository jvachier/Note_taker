"""Package each trained model in the layout published on the Hugging Face Hub.

Usage::

    uv run python -m src.publish.package --config configs/release.yaml

For every entry of ``models`` the seed with the best **validation** metric is chosen, its
checkpoint is re-saved without optimizer state (inference only), and
``<output_dir>/<repo name>/`` gets:

- ``model.keras``: the model (custom layers registered under ``dialogue_models``);
- ``tokenizer.json``: our WordPiece tokenizer;
- ``config.json``: what loading and inference need (kind, architecture, inference settings) and
  where the weights came from (run, seed, validation metric);
- ``vocab.txt`` and ``results.md`` (the model's results table, if configured);
- everything in ``src/publish/hub/<repo name>/``: the model card (``README.md``), a
  self-contained ``modeling.py`` (custom layers + inference, no dependency on this repo) and
  ``example.py``.

The demo app (``app/``) reads this same layout from a local folder or from the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import keras

import src.model1_scorer.model  # noqa: F401  (registers the custom layers for loading)
import src.model2_cleanup.model  # noqa: F401
from src.config import load_config
from src.eval.human_eval import best_seed

log = logging.getLogger(__name__)
FORMAT_VERSION = 1
HUB_DIR = Path(__file__).parent / "hub"  # per-repo files uploaded as they are


def inference_settings(kind: str, run_cfg: dict[str, Any]) -> dict[str, Any]:
    """Settings the app needs to run the model like the evaluation did."""
    if kind == "cleanup":
        return {
            "window_words": run_cfg["data"]["window_words"],
            "max_len": run_cfg["data"]["max_len"],
        }
    if kind == "scorer":
        return {"max_len": run_cfg["model"]["max_len"], "top_k": run_cfg["inference"]["top_k"]}
    raise ValueError(f"unknown model kind: {kind}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_model(repo: str, entry: dict[str, Any], tokenizer_dir: Path, output_dir: Path) -> Path:
    """Write ``output_dir/repo`` for one model and return it."""
    run_cfg = load_config(Path(entry["config"]))
    results_dir = Path(run_cfg["paths"]["results_dir"])
    seed, value = best_seed(results_dir, run_cfg["name"], tuple(entry["metric"]))
    ckpt = Path(run_cfg["paths"]["output_dir"]) / f"seed_{seed}" / "best.keras"
    tokenizer = tokenizer_dir / "tokenizer.json"
    manifest = results_dir / run_cfg["name"] / f"seed_{seed}" / "manifest.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text()).get("tokenizer_sha256")
        if expected and expected != sha256(tokenizer):
            raise ValueError(f"{tokenizer} is not the tokenizer {run_cfg['name']} was trained with")

    out = output_dir / repo
    out.mkdir(parents=True, exist_ok=True)
    model = keras.saving.load_model(ckpt, compile=False)  # drops the optimizer state
    model.save(out / "model.keras")
    shutil.copyfile(tokenizer, out / "tokenizer.json")
    if (tokenizer_dir / "vocab.txt").exists():
        shutil.copyfile(tokenizer_dir / "vocab.txt", out / "vocab.txt")
    hub_files = HUB_DIR / repo  # hand-written: model card, modeling.py, example.py
    if hub_files.is_dir():
        shutil.copytree(
            hub_files, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
        )
    if entry.get("results_table"):
        shutil.copyfile(entry["results_table"], out / "results.md")
    config = {
        "format_version": FORMAT_VERSION,
        "kind": entry["kind"],
        "architecture": run_cfg["model"],
        "inference": inference_settings(entry["kind"], run_cfg),
        "source": {
            "run": run_cfg["name"],
            "seed": seed,
            "selected_on": ".".join(entry["metric"]),
            "validation_value": value,
            "params": int(model.count_params()),
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    size = (out / "model.keras").stat().st_size / 1e6
    log.info(
        "%s: %s seed %d (%s %.3f), model.keras %.1f MB",
        repo,
        run_cfg["name"],
        seed,
        ".".join(entry["metric"]),
        value,
        size,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    """Package every model listed in the release config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/release.yaml"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    keras.utils.set_random_seed(cfg["seed"])
    for repo, entry in cfg["models"].items():
        package_model(repo, entry, Path(cfg["tokenizer_dir"]), Path(cfg["output_dir"]))


if __name__ == "__main__":
    main()
