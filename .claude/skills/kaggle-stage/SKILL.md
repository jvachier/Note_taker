---
name: kaggle-stage
description: Turn one pipeline stage (tokenizer, Model 2 training, teacher labelling, student training, evaluation) into a Kaggle notebook that runs on GPU, checkpoints, resumes and saves its outputs. Use when asked to run, prepare or package a stage for Kaggle.
---

# Kaggle stage

Kaggle limits: 12 h per session, ~30 GPU-hours per week. Every notebook must survive being
cut off and resumed.

## One self-contained notebook per model
- `model2_cleanup.ipynb` (Model 2), `model1_teachers.ipynb` (Model 1 teachers and baselines)
  and `model1_notes.ipynb` (Model 1 students), all in `notebooks/kaggle/`. Model 1's MLM
  sentence encoder is an earlier Kaggle output used as is (its notebook is archived). Each runs its whole stage:
  data, arrays, training, evaluation, results table.
- **Not connected to this repo**: no `src` imports, no `git clone`, no embedded or generated
  repo code, no `configs/` files. All code is written in the notebook's own cells and every
  hyperparameter is in its `CONFIG` cell. Upload the `.ipynb`, set accelerator + Internet,
  Run All.
- The code is an independent copy of the logic in `src/`; when one changes, update the other
  by hand. The one hard contract is the saved model: custom layers registered under
  `dialogue_models` with the same class names and constructor arguments as `src/layers`, so the
  repo loads the `.keras` files (`tests/test_notebooks.py` checks this).
- Commit notebooks without outputs.
- Fail fast on inputs: check attachments and downloads in the first cells, before any GPU
  work. Batch sessions cannot attach models. Reuse partial outputs instead of rerunning a
  finished step.

## Notebook cells, in order
1. `import os; os.environ["KERAS_BACKEND"] = "jax"`, then `ON_KAGGLE`, `SMOKE` (env `SMOKE=1`:
   tiny CPU run), `WORK` / `INPUT` (env `WORK_DIR` / `INPUT_DIR` override for local runs).
2. `pip install` of exact versions (Kaggle only).
3. `CONFIG` dict: every hyperparameter, with smoke overrides.
4. Imports, environment print, fail loudly if no GPU on Kaggle.
5. Resume: walk `/kaggle/input` (`os.walk(..., followlinks=True)`) for a previous output of this
   notebook and copy its `data/`, `checkpoints/`, `results/` back; skip finished steps.
6. Sections with their code and a short markdown explanation each.
7. Outputs in `/kaggle/working/`: `checkpoints/<model>/seed_<n>/best.keras`,
   `checkpoints/<model>/tokenizer/`, `results/<name>/seed_<n>/`, `results/tables/<name>.md`,
   `manifest_<notebook>.json`.

## Training requirements (in the notebook, configured in `CONFIG`)
- `keras.callbacks.BackupAndRestore(backup_dir=...)` so a restarted session resumes.
- `keras.callbacks.ModelCheckpoint` every epoch, keep the best on the validation metric.
- A time-guard callback that stops training cleanly at `max_hours` (default 11).
- Mixed precision `mixed_float16` on T4. On 2 GPUs (T4 x2) use
  `keras.distribution.set_distribution(keras.distribution.DataParallel())`.
- Log per-epoch metrics to CSV (`keras.callbacks.CSVLogger`).

## Teacher labelling (Model 1, special case)
- The teachers run inside the Model 1 notebook but in **their own virtual environment**: the
  notebook writes the labelling script to disk, builds a venv with vLLM, and runs it as a
  subprocess **before** the notebook imports JAX (so JAX has not claimed GPU memory). With two
  GPUs, one teacher per GPU in parallel (`CUDA_VISIBLE_DEVICES`). The training kernel never
  imports vLLM or PyTorch. In smoke mode a deterministic fake teacher replaces Qwen.
- A labelling deadline leaves time for the other phases of the session.
- v3 (`model1_teachers.ipynb`): Qwen3-0.6B in thinking mode, plus RoBERTa-Large and DistilBERT
  fine-tuned with KerasHub on JAX (no PyTorch). The venv is built with uv (`python -m venv`
  fails on Kaggle).
- Write labels incrementally (JSONL, one dialogue per line) and skip dialogues already labelled,
  so a cut-off session resumes where it stopped.
- If vLLM fails to install, fall back to `transformers` + `bitsandbytes` 4-bit and say so.

## Bringing outputs back to the repo
The repo uses the trained models (CPU evaluation, results tables, `hf-publish`). After a Kaggle
run: `uv run python notebooks/kaggle/fetch_outputs.py <notebook>` downloads the latest output
(`kaggle kernels output`) and copies `checkpoints/<model>/` (without `backup/`) and `results/`
to the same paths here; the manifest goes to `results/kaggle/`. Add new notebooks to `PULL`.

## Before uploading
- `XLA_FLAGS=--xla_force_host_platform_device_count=2 RUN_NOTEBOOKS=1 uv run pytest
  tests/test_notebooks.py`: executes the notebooks in smoke mode on CPU **with 2 simulated
  devices** (the same data-parallel batch splitting as Kaggle's T4 x2, so uneven batches fail
  here, not on Kaggle) and loads the outputs with `src/`.
- With data parallelism every batch must split evenly: datasets pad a partial last batch with
  zero-weight rows, and prediction pads to whole batches and drops the padding.
- `uv run pytest -q` and lint pass.
- Report: inputs to attach (none, or the previous output to resume), accelerator, expected
  GPU time.
