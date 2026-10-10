---
name: hf-publish
description: Package a trained Keras model (Model 2 cleanup or Model 1 utterance scorer) with its tokenizer, config and model card, and publish it to the Hugging Face Hub. Use when asked to publish, upload or release a model.
---

# Publish a model on Hugging Face

## Preconditions (stop and tell the user if any fails)
- Final results exist in `results/tables/` for this model, with all seeds.
- `uv run pytest -q` passes.
- The chosen checkpoint is the best on **validation**, and its test metrics are in the table.

## Repo
- Hub repo: `Jvachier/<model-name>` (e.g. `dialogue-cleanup-7m`, `dialogue-notes-scorer-17m`).
  Confirm the name with the user before creating it.
- Login via `HF_TOKEN` environment variable or `huggingface-cli login`. Never write the token to a file in the repo.

## Folder to upload (`release/<model-name>/`)
- `model.keras`: the final model
- `config.json`: architecture hyperparameters (vocab, d_model, layers, heads, FFN, max length, heads)
- `tokenizer.json` and `vocab.txt`: our WordPiece tokenizer
- `README.md`: the model card (below)
- `results.md`: the results table for this model
- `example.py`: load and run on one example

Custom layers are registered under `package="dialogue_models"`, so loading needs the code:
`pip install git+https://github.com/<user>/<repo>@<release_tag>`. Tag the GitHub release first
and use that tag in the card and in `example.py`.

## Model card (`README.md`)
YAML header: `library_name: keras`, `language: en`, `license`, `datasets`, `metrics`, `tags`
(`keras`, `jax`, `from-scratch`, `knowledge-distillation` for Model 1, `punctuation-restoration` for Model 2).

Sections:
1. **Summary**: what it does, params, size on disk, CPU latency on M2.
2. **Built from scratch**: own tokenizer, own architecture, Keras 3 + JAX; for Model 1, MLM pretraining and distillation from Qwen2.5 teachers.
3. **Architecture**: table from `config.json`.
4. **Training data**: datasets, splits, noise process (Model 2), teacher prompt and label validation (Model 1).
5. **Results**: table with baselines and teachers, mean ± std over seeds.
6. **Usage**: install line + short snippet from `example.py`, with a before/after example.
7. **Limitations**: English only; domain (WikiText, DialogSum); Model 2 does not fix misrecognised words; Model 1 is extractive and its tags reflect agreement with the teacher, not human labels.
8. **Licenses**: list the license of every dataset and teacher. DialogSum is non-commercial,
   so the model license must be compatible; check the dataset licenses and confirm the final license with the user before uploading.

## Upload
- `huggingface_hub.HfApi().create_repo(..., exist_ok=True)` then `upload_folder(...)`.

## Verify after upload
- In a fresh virtual environment: install the tagged repo, download the files with
  `huggingface_hub.snapshot_download`, load `model.keras`, run `example.py`, and check the output
  matches the local run.
- Report the Hub URL and the verification result.
