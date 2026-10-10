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
```

Configs live in `configs/`. A config can inherit from another with `base: <file>.yaml`;
`src.config.load_config` returns the fully resolved dict.
