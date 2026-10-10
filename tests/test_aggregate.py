"""Tests for the results aggregation script."""

import json

import pytest

from src.eval.aggregate import main, mean_std


def _report(f1):
    pc = {n: {"f1": f1} for n in ("comma", "period", "question", "capitalised", "upper")}
    return {"punct": {"macro_f1": f1, "per_class": pc}, "casing": {"accuracy": f1, "per_class": pc}}


def test_mean_std():
    assert mean_std([0.5]) == "0.500"
    assert mean_std([0.4, 0.6]) == "0.500 ± 0.141"


def test_model2_table(tmp_path):
    for seed, f1 in ((0, 0.8), (1, 0.6)):
        metrics = {
            "seed": seed,
            "params": 7_000_000,
            "validation": _report(f1),
            "test": {"test_wikitext_noise10": {"noise": 0.1, "baseline": _report(0.3),
                                               "model": _report(f1)}},
        }  # fmt: skip
        path = tmp_path / "exp" / f"seed_{seed}" / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(metrics))
    main(["--exp", "exp", "--results-dir", str(tmp_path)])
    table = (tmp_path / "tables" / "exp.md").read_text()
    assert "| wikitext | 10% | Rule baseline | 0.300 |" in table
    assert "| wikitext | 10% | Model (exp) | 0.700 ± 0.141 |" in table
    assert "7,000,000" in table


def test_missing_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--exp", "nope", "--results-dir", str(tmp_path)])
