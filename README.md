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
uv run python -m src.model1_scorer.oracle --config configs/oracle.yaml         # Stage 6b: oracle labels
uv run python -m src.model1_scorer.data --config configs/model1_student_A.yaml # Stage 7a: student arrays
uv run python -m src.model1_scorer.train --config configs/model1_student_A.yaml --all-seeds  # 7b (A-D)
uv run --extra eval python -m src.model1_scorer.evaluate --config configs/model1_eval.yaml  # 7c: table
```

Teacher labels (Stage 6a) are produced on Kaggle only, in a vLLM environment:
`python -m src.model1_scorer.teacher.label --config configs/teacher.yaml --teacher qwen7b`.

Smoke tests (CPU, < 2 min): the Model 2 commands with `configs/model2_tiny.yaml`, the MLM
commands with `configs/mlm_tiny.yaml`, the student commands with
`configs/model1_student_tiny.yaml` (trains on oracle labels, since teacher labels exist only on
Kaggle).
Full runs happen on Kaggle with one self-contained notebook per model (upload it, GPU T4 x2,
Internet on, Run All; attach its previous output to resume). They do not use this repo's code.
- `notebooks/kaggle/model2_cleanup.ipynb`: data, tokenizer, training, evaluation (1 session).
- `notebooks/kaggle/model1_scorer.ipynb`: data, teacher labels (own vLLM environment), oracle,
  MLM pretraining, students A-D, preview table (2-3 sessions). Attach the Model 2 output too so
  both models share the exact same tokenizer file.

Bring the outputs back with `uv run python notebooks/kaggle/fetch_outputs.py <notebook>`, then
for Model 1 build the local test arrays and run the full evaluation (BERTScore, CPU latency):
`uv run python -m src.model1_scorer.data --config configs/model1_student_A.yaml` and
`uv run --extra eval python -m src.model1_scorer.evaluate --config configs/model1_eval.yaml`.

Configs live in `configs/`. A config can inherit from another with `base: <file>.yaml`;
`src.config.load_config` returns the fully resolved dict.

## Results report

```bash
uv run python -m src.eval.report    # -> results/report/index.html (interactive, offline)
```

Every figure comes with what it shows and a good / mixed / poor verdict computed from the
numbers (thresholds in `configs/report.yaml`).

## Release and demo app

```bash
uv run python -m src.publish.package   # best validation seed of each model -> release/<repo>/
```

Each `release/<repo>/` folder (`model.keras` without optimizer state, `tokenizer.json`,
`config.json`) is what gets uploaded to the Hugging Face Hub, and what the demo app loads.

Run the demo (Dash: punctuation restoration + dialogue notes) with Docker:

```bash
docker compose up --build                       # models from the Hugging Face Hub
MODEL2_SOURCE=/models/punctuation-casing-en-7m \
MODEL1_SOURCE=/models/dialogue-notes-scorer-17m \
docker compose up --build                       # or the local ./release folders
```

then open http://localhost:8050. Without Docker: `uv run --extra app python -m app.main` (uses
`./release` if present, else the Hub; override with `MODEL2_SOURCE` / `MODEL1_SOURCE`).

## CI

`.github/workflows/ci.yaml` runs `pre-commit run --all-files` (YAML/TOML checks, large-file
guard, ruff, ruff format, pytest) on every pull request and push to `main`, and checks that the
Docker image builds. Install the hooks locally with `uv run pre-commit install`.
