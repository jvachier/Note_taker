---
name: kaggle-stage
description: Turn one pipeline stage (tokenizer, Model 2 training, MLM pretraining, teacher labelling, student training, evaluation) into a Kaggle notebook that runs on GPU, checkpoints, resumes and saves its outputs. Use when asked to run, prepare or package a stage for Kaggle.
---

# Kaggle stage

Kaggle limits: 12 h per session, ~30 GPU-hours per week. Every notebook must survive being
cut off and resumed.

## Files to create
- `notebooks/kaggle/<NN>_<stage>/notebook.ipynb` (thin: no logic, only calls into `src/`)
- `notebooks/kaggle/<NN>_<stage>/kernel-metadata.json` for `kaggle kernels push`
  (`enable_gpu: true`, `enable_internet: true`, dataset sources listed in `dataset_sources`)

## Notebook cells, in order
1. `import os; os.environ["KERAS_BACKEND"] = "jax"` (before any keras import).
2. Install the repo pinned to a commit:
   `pip install -q "git+https://github.com/<user>/<repo>@<commit_sha>"`
3. Print environment: `jax.devices()`, keras version, commit SHA. Fail loudly if no GPU
   (except for CPU-only stages).
4. Load the config: `configs/<stage>.yaml`. Inputs come from `/kaggle/input/<dataset>/`.
5. Run the stage entry point, e.g. `python -m src.model2_cleanup.train --config ...`.
6. Outputs go to `/kaggle/working/<stage>/`:
   - checkpoints, final `.keras` model, metrics JSON, logs
   - `manifest.json`: commit SHA, config (full copy), input dataset versions, device, start/end
     time, number of steps completed.

## Training requirements (implemented in `src/`, configured in YAML)
- `keras.callbacks.BackupAndRestore(backup_dir=...)` so a restarted session resumes.
- `keras.callbacks.ModelCheckpoint` every epoch, keep the best on the validation metric.
- A time-guard callback that stops training cleanly at `max_hours` (default 11).
- Mixed precision `mixed_float16` on T4. On 2 GPUs (T4 x2) use
  `keras.distribution.set_distribution(keras.distribution.DataParallel())`.
- Log per-epoch metrics to CSV (`keras.callbacks.CSVLogger`).

## Teacher labelling stage (special case)
- Separate notebook, **no JAX/Keras installed**: vLLM only.
- Model: `Qwen/Qwen2.5-7B-Instruct-AWQ` or `Qwen/Qwen2.5-1.5B-Instruct`, temperature 0.
- Write labels incrementally (JSONL, one dialogue per line) and skip dialogues already labelled,
  so a cut-off session resumes where it stopped.
- If vLLM fails to install, fall back to `transformers` + `bitsandbytes` 4-bit and say so.

## Hand-over to the next stage
Tell the user to save the notebook output as a new version of a Kaggle Dataset named
`<stage>-outputs`, and add that dataset to the next stage's `dataset_sources`.

## Before pushing
- Run the same entry point locally with `configs/<stage>_tiny.yaml` on CPU; it must finish.
- `uv run pytest -q` passes.
- Report: commit SHA, dataset names, expected GPU time, and the exact push command.
