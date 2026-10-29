# `kaggle-stage`: what this skill does

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills): instructions Claude
loads when you ask it to prepare a pipeline stage for Kaggle. Claude follows
[`SKILL.md`](SKILL.md); this file explains it for people.

## Purpose

Real training runs on Kaggle's free GPUs (T4, 12 hours per session, about 30 GPU-hours per
week). The skill turns a pipeline stage into a **self-contained Kaggle notebook**. You upload
it, set the accelerator and Internet, and click Run All. If the session is cut off, the
notebook resumes where it stopped.

The notebooks in [`notebooks/kaggle/`](../../../notebooks/kaggle/):

| Notebook | What it runs |
|---|---|
| `model2_cleanup.ipynb` | Model 2: data, tokenizer, training (3 seeds), evaluation |
| `model1_scorer.ipynb` | Model 1 v1/v2 (superseded) and the MLM pretraining of the sentence encoder used by v3 |
| `model1_teachers.ipynb` | Model 1 v3 teachers: Qwen3-0.6B labels (own vLLM environment), fine-tuned RoBERTa-Large and DistilBERT |
| `model1_notes.ipynb` | Model 1 v3 students: E1 (gold only, released) and E4 (with teachers), 3 seeds each |

## When it runs

When you ask Claude to "run", "prepare" or "package" a stage for Kaggle, or type
`/kaggle-stage`.

## What it does

1. **Writes the notebook with no link to this repo**: no `src` imports and no `git clone`.
   The code is an independent copy of the `src/` logic, and every hyperparameter sits in one
   `CONFIG` cell. The one hard contract is the saved model: the layer classes must match
   `src/layers` so the repo can load the `.keras` files.
2. **Makes it resumable**:
   - it finds a previous output attached under `/kaggle/input` and copies it back;
   - `BackupAndRestore` and `ModelCheckpoint` save training state every epoch;
   - a time guard stops cleanly before the 12-hour limit.
3. **Keeps teachers isolated**: language-model teachers run in their own virtual environment
   (vLLM), as a subprocess, before JAX starts. The training code never imports PyTorch or
   vLLM.
4. **Tests it locally** before upload: `RUN_NOTEBOOKS=1 uv run pytest tests/test_notebooks.py`
   runs each notebook in smoke mode on CPU, with two simulated devices to mimic Kaggle's
   T4 x2, and loads its outputs with `src/`.
5. **Tells you** what to attach (nothing, or the previous output to resume), which
   accelerator to pick, and the expected GPU time.

After a run, `uv run python notebooks/kaggle/fetch_outputs.py <notebook>` copies the
checkpoints and results back into the repo.

## Guardrails

- Notebooks are committed without outputs.
- When `src/` changes, the notebook copy must be updated by hand; the notebook tests catch
  layer mismatches.
- It checks inputs and downloads first and fails fast, so no GPU time is wasted on a missing
  attachment.

## Example request

> "Prepare the student training for Kaggle so it resumes if the session ends."

## Related

- [`run-experiment`](../run-experiment/explanations.md) defines the experiment and its config
  before this skill runs it.
- [`hf-publish`](../hf-publish/explanations.md) publishes the trained model.
