"""Aggregate per-seed metrics into a Markdown results table (mean ± std over seeds).

Usage::

    uv run python -m src.eval.aggregate --exp model2_cleanup

Reads ``results/<exp>/seed_*/metrics.json`` and writes ``results/tables/<exp>.md``.
Model 2 tables have one block per test set and noise level, with the rule baseline row
(deterministic, so it has no std) above the model row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL2_COLUMNS = [
    ("Punct macro-F1", ("punct", "macro_f1")),
    ("Comma F1", ("punct", "per_class", "comma", "f1")),
    ("Period F1", ("punct", "per_class", "period", "f1")),
    ("Question F1", ("punct", "per_class", "question", "f1")),
    ("Casing acc", ("casing", "accuracy")),
    ("Capitalised F1", ("casing", "per_class", "capitalised", "f1")),
    ("UPPER F1", ("casing", "per_class", "upper", "f1")),
]


def _get(d: dict[str, Any], path: tuple[str, ...]) -> float:
    for key in path:
        d = d[key]
    return float(d)


def mean_std(values: list[float]) -> str:
    """``"mean ± std"`` (sample std) with 3 decimals; a single value has no ``±``."""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def load_seeds(exp_dir: Path) -> dict[int, dict[str, Any]]:
    """``{seed: metrics}`` for every completed seed of an experiment."""
    seeds = {}
    for path in sorted(exp_dir.glob("seed_*/metrics.json")):
        seeds[int(path.parent.name.split("_")[1])] = json.loads(path.read_text())
    if not seeds:
        raise FileNotFoundError(f"No seed_*/metrics.json under {exp_dir}")
    return seeds


def model2_table(exp: str, seeds: dict[int, dict[str, Any]]) -> str:
    """Markdown table for a Model 2 experiment."""
    header = ["Test set", "Noise", "System", *[c for c, _ in MODEL2_COLUMNS]]
    lines = [
        f"# {exp}",
        "",
        f"Seeds: {sorted(seeds)} (mean ± std over seeds; the rule baseline is deterministic).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    first = next(iter(seeds.values()))
    for set_name, entry in first["test"].items():
        dataset = set_name.split("_")[1]
        noise = f"{entry['noise']:.0%}"
        baseline = [f"{_get(entry['baseline'], p):.3f}" for _, p in MODEL2_COLUMNS]
        lines.append(f"| {dataset} | {noise} | Rule baseline | " + " | ".join(baseline) + " |")
        model = [
            mean_std([_get(m["test"][set_name]["model"], p) for m in seeds.values()])
            for _, p in MODEL2_COLUMNS
        ]
        lines.append(f"| {dataset} | {noise} | Model ({exp}) | " + " | ".join(model) + " |")
    val = mean_std([m["validation"]["punct"]["macro_f1"] for m in seeds.values()])
    params = first["params"]
    lines += ["", f"Validation punct macro-F1: {val}. Parameters: {params:,}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Write ``results/tables/<exp>.md`` for one experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    seeds = load_seeds(args.results_dir / args.exp)
    if "baseline" not in next(iter(next(iter(seeds.values()))["test"].values())):
        raise NotImplementedError("Only Model 2 tables are implemented so far.")
    out = args.results_dir / "tables" / f"{args.exp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model2_table(args.exp, seeds))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
