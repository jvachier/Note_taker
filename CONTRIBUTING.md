# Contributing

Thanks for your interest in **Note_taker**. This repository trains two small models from
scratch and publishes them, so the bar for contributions is reproducibility and honest
evaluation, not just working code.

## Ground rules

- [PLAN.md](PLAN.md) is the source of truth for data, tokenizer, architectures, losses,
  hyperparameters, metrics and tests. Open an issue before proposing a change to it; every
  deviation is recorded in its progress log.
- **Keras 3 on the JAX backend.** No PyTorch and no TensorFlow in `src/` or `app/` (no
  `tf.data`, no `tensorflow_text`). `tests/test_import_rules.py` enforces this.
- **From scratch.** No pretrained weights and no pretrained tokenizers in our models. The
  WordPiece tokenizer is trained here with the `tokenizers` library.
- **Teachers and baselines** (Qwen, RoBERTa, DistilBERT) run only in the Kaggle teacher
  notebook. Only their cached outputs reach training, and their weights are never part of a
  published model.
- **Custom layers** use `keras.ops`, are registered with
  `@keras.saving.register_keras_serializable(package="dialogue_models")` and implement
  `get_config`.
- **Configs, not constants.** Every hyperparameter lives in `configs/*.yaml`. Every run takes
  a `seed` and sets it for Python, NumPy and Keras.
- **Splits.** Official DialogSum and WikiText splits; the meeting corpus's own splits. No
  dialogue or meeting ID in more than one split. Never tune on test.
- Never commit data, checkpoints, release folders or tokens. They are git-ignored; the
  pre-commit large-file guard stops anything over 1 MB.

## Setup

Requirements: Python 3.11 or 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # add --extra app for the demo app
uv run pre-commit install
```

## Before opening a pull request

```bash
uv run pytest -q
uv run ruff check . --fix && uv run ruff format .
```

Both must pass. CI runs `pre-commit run --all-files` (YAML/TOML checks, large-file guard,
ruff, ruff format, pytest) on Python 3.11.

## Code contributions

- Keep modules small and readable, with type hints and docstrings. Every new module gets
  tests in `tests/`. Tests run offline on CPU, with tiny models and no downloads.
- New experiments get their own config inheriting from an existing one (`base: <file>.yaml`)
  and a `*_tiny.yaml` smoke config that runs in under two minutes on CPU.
- `src/` is the only training code in the repository. Kaggle notebooks are local working
  copies and are git-ignored; never commit them. Their outputs come back with
  `notebooks/kaggle/fetch_outputs.py`.
- The Hub files in `src/publish/hub/<model>/` (model card, `modeling.py`, `example.py`) ship
  with the published models. `modeling.py` must give the same outputs as the repo code.
- Follow the existing style (ruff, line length 100).

## Results

- Report every metric as mean ± std over seeds, with the baselines in the same table.
- Never overwrite a results folder; a rerun gets a new experiment name.
- A change that moves published numbers needs the updated table in `results/tables/` and a
  note in PLAN.md's progress log.

## Commits and pull requests

- Use short, imperative commit subjects ("Add balanced zone calibration").
- Keep pull requests focused: one stage, one experiment or one fix.
- Every change needs review from the code owner (see
  [`.github/CODEOWNERS`](.github/CODEOWNERS)).

## Licence

By contributing, you agree that your contributions are licensed under the MIT licence
([LICENSE](LICENSE)).
