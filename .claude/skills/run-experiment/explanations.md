# `run-experiment`: what this skill does

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills): instructions Claude
loads when you ask it to train, compare or ablate. Claude follows [`SKILL.md`](SKILL.md); this
file explains it for people.

## Purpose

Run every experiment the same way: a versioned config, several seeds, a smoke test first, and
results recorded as **mean ± std** next to the baselines. Two runs can then be compared, and a
difference counts only if it is larger than the spread across seeds.

Examples from this project:
- the Model 2 noise levels;
- the v3 runs E1 (gold labels only) and E4 (gold plus teacher distillation).

## When it runs

When you ask Claude to "train", "compare", "ablate" or "add a row to the results", or type
`/run-experiment`.

## What it does

1. **Config**: creates `configs/<exp_name>.yaml`. It inherits from the closest existing config
   (`base: <parent>.yaml`), overrides only what the experiment tests, and gets a `description:`
   stating the question it answers. A `<exp_name>_tiny.yaml` smoke config comes with it.
2. **Smoke test**: runs the tiny config end to end on CPU, in under two minutes. The loss must
   go down, and the metrics file must be written.
3. **Full run**: on Kaggle, through [`kaggle-stage`](../kaggle-stage/explanations.md).
4. **Records results**:
   - each seed writes `results/<exp_name>/seed_<n>/metrics.json` and `manifest.json` (commit,
     resolved config, device, time);
   - the results table is rebuilt: `src.eval.aggregate` for Model 2, and
     `src.model1_notes.evaluate` for Model 1 v3 (`results/tables/model1_v3.md`).
5. **Reports**: the question, the numbers against the baselines, whether the difference
   exceeds the seed std, and the next experiment worth running.

## Guardrails

- Never tunes on the test split: thresholds, early stopping and checkpoint choice use
  validation only.
- Never overwrites a results folder; a rerun gets a new name (for example `_v2`).
- Every table keeps its baseline rows (rule baseline, Lead-k, teachers).

## Example request

> "Run E4 again without the Qwen losses and compare it with E1."

## Related

- [`kaggle-stage`](../kaggle-stage/explanations.md) runs the full training on GPU.
- [`hf-publish`](../hf-publish/explanations.md) releases the best run.
