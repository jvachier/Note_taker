# CLAUDE.md

Two small transformer models built **from scratch** and published on Hugging Face:
- **Model 2 – Cleanup**: restores punctuation and casing in noisy lowercase text.
- **Model 1 – Meeting notes** (v3, PLAN.md §10): hierarchical student that extracts decisions,
  action items and key points from meetings. Trained on the gold-labelled corpus
  `meeting-notes-extractive-v1` (separate repo) and distilled from Qwen3-0.6B's reasoning and a
  fine-tuned RoBERTa-Large. The v1/v2 DialogSum scorer (`src/model1_scorer/`) is kept as history.

`PLAN.md` is the source of truth for data, tokenizer, architectures, losses, hyperparameters,
metrics and tests. Read it before writing code. If something here conflicts with PLAN.md, ask.

## How to work
- Work **one stage at a time**, following "Order of work" in PLAN.md.
- At the end of a stage: run tests and lint, then stop and report what was built, what was
  tested, and any deviation from PLAN.md. Do not start the next stage without a go-ahead.
- Any deviation from PLAN.md (architecture, data, metric, library) needs my approval first.
- Prefer small, readable modules with type hints and docstrings. Every new module gets tests.

## Hard rules
- **Keras 3 with the JAX backend.** Set `KERAS_BACKEND=jax` before importing keras.
- **No PyTorch. No TensorFlow** in `src/` (no `tf.data`, no `tensorflow_text`).
- Custom layers use `keras.ops` only. `keras.layers.MultiHeadAttention`, `Dense`,
  `LayerNormalization`, `Dropout`, `Embedding` are allowed.
- Register every custom layer/model with
  `@keras.saving.register_keras_serializable(package="dialogue_models")` and implement `get_config`.
- **No pretrained weights or pretrained tokenizers** in our models. The tokenizer is trained by us
  with the `tokenizers` library (WordPiece).
- **Teachers and baselines** (Qwen3-0.6B, RoBERTa-Large, DistilBERT) run **only on Kaggle, only in
  the Model 1 teacher notebook** (Notebook 1), never in the student training environment:
  - Qwen runs in its own virtual environment (vLLM). Reference code lives in
    `src/model1_notes/teacher/` (and `src/model1_scorer/teacher/` for v1/v2), tested locally
    with a fake teacher.
  - RoBERTa-Large and DistilBERT are fine-tuned there with KerasHub on the JAX backend (no
    PyTorch).
  - Only their cached outputs (`.npy` / Parquet) reach Notebook 2. Their weights are never
    part of our published models.
- Never download Qwen, RoBERTa-Large or DistilBERT locally.
- BERTScore and ROUGE are evaluation-only; never import them in training code.

## Data and configs
- Raw and processed data live in `data/` (git-ignored). Never commit data, checkpoints or tokens.
- Pre-tokenize to `int32` NumPy arrays (`.npy`, memory-mapped). Training reads arrays, not text.
- Every hyperparameter lives in `configs/*.yaml`. No hard-coded hyperparameters in `src/`.
- Every run takes a `seed` and sets it for Python, NumPy and Keras (`keras.utils.set_random_seed`).
- Splits: official DialogSum and WikiText splits; for Model 1 v3, the corpus's own splits
  (pinned version). No dialogue or meeting ID in more than one split.

## Local machine (MacBook M2, 16 GB RAM)
- Local = development, tests, tiny smoke runs, CPU evaluation. Real training runs on Kaggle GPUs.
- Keep local data under ~2 GB. Smoke configs (`configs/*_tiny.yaml`) must run in under 2 minutes on CPU.

## Commands
- Environment: `uv sync`
- Tests: `uv run pytest -q`
- Lint/format: `uv run ruff check . --fix && uv run ruff format .`
- A stage is done only when tests pass and lint is clean.

## Layout
```
configs/  src/data/  src/tokenizer/  src/layers/  src/model2_cleanup/
src/model1_scorer/  src/model1_notes/  src/eval/  notebooks/kaggle/  results/  tests/
```

## Skills in this repo (`.claude/skills/`)
- `kaggle-stage`: turn a stage into a Kaggle notebook that checkpoints and resumes.
- `run-experiment`: create a config, run seeds, record results.
- `hf-publish`: package a trained model and publish it on Hugging Face.
