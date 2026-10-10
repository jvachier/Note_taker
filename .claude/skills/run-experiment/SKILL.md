---
name: run-experiment
description: Create and run a training or evaluation experiment (Model 2, MLM pretraining, students A/B/C/D, ablations) with a versioned config and multiple seeds, then record mean ± std in the results tables. Use when asked to train, compare, ablate or add a row to the results.
---

# Run an experiment

## 1. Config
- Create `configs/<exp_name>.yaml` (flat directory) that inherits from the closest existing
  config with `base: <parent>.yaml` and sets a new `name:`.
  Naming: `<model>_<variant>`, e.g. `model1_student_A`, `model1_student_D_noMLM`, `model2_noise10`.
- Change only what the experiment tests: list only the overridden keys (nested keys are
  deep-merged). Add a `description:` field stating the question it answers.
- `src.config.load_config` resolves inheritance; save that resolved dict in `manifest.json`.
- Required fields: `seeds: [0, 1, 2]`, data paths, model dims, optimizer, schedule, batch size,
  epochs, early-stopping metric, `max_hours`.
- Also create `<exp_name>_tiny.yaml` (tiny model, 50 steps) for the local smoke test.

## 2. Smoke test (local, CPU)
- Run the tiny config end to end, including evaluation. It must finish in under 2 minutes.
- Check that the loss decreases and that metrics JSON is written.

## 3. Full run
- On Kaggle via the `kaggle-stage` skill. One seed per run or all seeds in sequence,
  whichever fits within `max_hours`.

## 4. Record results
- Each seed writes `results/<exp_name>/seed_<n>/metrics.json` and `manifest.json`
  (commit SHA, config copy, data versions, device, wall-clock time).
- Run `python -m src.eval.aggregate --exp <exp_name>` to update:
  - `results/tables/model2.md` or `results/tables/model1.md`
  - each cell as `mean ± std` over seeds
- Model 1 rows must include: params, ROUGE-1/2/L, BERTScore F1, retention vs 7B teacher,
  Spearman vs 7B teacher, top-3 Jaccard, decision/action-item F1, CPU latency on M2.
- Model 2 rows: punctuation per-class F1 and macro-F1, casing accuracy and F1,
  on WikiText test and DialogSum test, at each noise level.

## Rules
- Never overwrite an existing results folder; a rerun gets a new `exp_name` suffix (`_v2`).
- Never tune on the test split. Model selection and early stopping use validation only.
- Every table keeps its baseline rows (rule baseline for Model 2; lead-3, oracle and teachers for Model 1).

## Report to the user
Short summary: the question, the numbers vs the baseline/teacher, whether the difference is larger
than the seed std, and the next experiment it suggests.
