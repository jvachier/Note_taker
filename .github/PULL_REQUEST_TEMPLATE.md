## What and why

<!-- What does this change, and why? Link the issue it closes (Closes #...). -->

## Type

- [ ] Bug fix
- [ ] New experiment or results (configs, tables)
- [ ] Model / training / data pipeline change
- [ ] Demo app
- [ ] Release or model card (Hugging Face files in `src/publish/hub/`)
- [ ] Docs, CI or tooling

## Checklist

- [ ] `uv run pytest -q` passes.
- [ ] `uv run ruff check . --fix && uv run ruff format .` leaves nothing to fix.
- [ ] New modules have type hints, docstrings and tests.
- [ ] Hyperparameters are in `configs/*.yaml`, not in code. Every run takes a `seed`.
- [ ] No PyTorch or TensorFlow in `src/` or `app/`; no pretrained weights in our models.
- [ ] No data, checkpoints, release folders, tokens or account names committed (Hub
      repos are written `<hf-user>/...`).
- [ ] Deviations from `PLAN.md` are approved and logged in its progress log.

If relevant:

- [ ] Results reported as mean ± std over seeds, with baselines; the table in
      `results/tables/` is regenerated. Model selection used validation only.
- [ ] Hub `modeling.py` gives the same outputs as the repo code.

## Results or screenshots

<!-- Numbers before / after (mean ± std), or a screenshot of the app. Delete if not relevant. -->
