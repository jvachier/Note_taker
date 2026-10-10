# CLAUDE.md

Two small transformer models built **from scratch** and published on Hugging Face:
- **Model 2 – Cleanup**: restores punctuation and casing in noisy lowercase text.
- **Model 1 – Utterance scorer**: extractive dialogue notes, distilled from Qwen teachers.

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
- The Qwen teachers are used **only** to produce labels, only on Kaggle, and never in the same
  environment as training: in `src/model1_scorer/teacher/` (reference code, tested locally with a
  fake teacher) and in the Model 1 notebook, where they run in their own virtual environment
  (vLLM) as a subprocess before JAX starts.
- BERTScore and ROUGE are evaluation-only; never import them in training code.

## Data and configs
- Raw and processed data live in `data/` (git-ignored). Never commit data, checkpoints or tokens.
- Pre-tokenize to `int32` NumPy arrays (`.npy`, memory-mapped). Training reads arrays, not text.
- Every hyperparameter lives in `configs/*.yaml`. No hard-coded hyperparameters in `src/`.
- Every run takes a `seed` and sets it for Python, NumPy and Keras (`keras.utils.set_random_seed`).
- Splits: official DialogSum and WikiText splits; no dialogue ID in more than one split.

## Local machine (MacBook M2, 16 GB RAM)
- Local = development, tests, tiny smoke runs, CPU evaluation. Real training runs on Kaggle GPUs.
- Never download the teacher models locally.
- Keep local data under ~2 GB. Smoke configs (`configs/*_tiny.yaml`) must run in under 2 minutes on CPU.

## Commands
- Environment: `uv sync`
- Tests: `uv run pytest -q`
- Lint/format: `uv run ruff check . --fix && uv run ruff format .`
- A stage is done only when tests pass and lint is clean.

## Layout
```
configs/  src/data/  src/tokenizer/  src/layers/  src/model2_cleanup/
src/model1_scorer/  src/eval/  notebooks/kaggle/  results/  tests/
```

## Skills in this repo (`.claude/skills/`)
- `kaggle-stage`: turn a stage into a Kaggle notebook that checkpoints and resumes.
- `run-experiment`: create a config, run seeds, record results.
- `hf-publish`: package a trained model and publish it on Hugging Face.
