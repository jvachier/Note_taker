# Note_taker

Two small transformer models built from scratch with Keras 3 (JAX backend):

- **Model 2 – Cleanup**: restores punctuation and casing in noisy lowercase text.
- **Model 1 – Utterance scorer**: extractive dialogue notes, distilled from Qwen teachers.

See [PLAN.md](PLAN.md) for the full design.

```bash
uv sync                    # training/dev environment (no PyTorch, no TensorFlow)
uv sync --extra eval       # + BERTScore (pulls PyTorch; evaluation only)
uv run pytest -q
uv run ruff check . --fix && uv run ruff format .
```

Pipeline stages (local; outputs under the git-ignored `data/`):

```bash
uv run python -m src.data.download --config configs/data.yaml        # Stage 2: data + splits
uv run python -m src.tokenizer.train --config configs/tokenizer.yaml # Stage 3: WordPiece
uv run python -m src.model2_cleanup.build_data --config configs/model2.yaml   # Stage 4a: arrays
uv run python -m src.model2_cleanup.train --config configs/model2.yaml --all-seeds  # Stage 4b
uv run python -m src.eval.aggregate --exp model2_cleanup                     # results table
uv run python -m src.model1_scorer.mlm_data --config configs/mlm.yaml          # Stage 5a: MLM arrays
uv run python -m src.model1_scorer.mlm_train --config configs/mlm.yaml         # Stage 5b: pretrain
```

Smoke tests (CPU, < 2 min): the Model 2 commands with `configs/model2_tiny.yaml`, the MLM
commands with `configs/mlm_tiny.yaml`.
Full runs happen on Kaggle with one self-contained notebook per model:
`notebooks/kaggle/model2_cleanup.ipynb` (upload it, GPU T4 x2, Internet on, Run All; attach its
previous output to resume). It does not use this repo's code. Bring the trained models and
results back with `uv run python notebooks/kaggle/fetch_outputs.py model2_cleanup`
(into `checkpoints/model2/` and `results/`).

Configs live in `configs/`. A config can inherit from another with `base: <file>.yaml`;
`src.config.load_config` returns the fully resolved dict.
