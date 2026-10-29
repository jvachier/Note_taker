---
name: hf-publish
description: Package a trained Keras model (Model 2 cleanup or Model 1 meeting notes) with its tokenizer, config and model card, and publish it to the Hugging Face Hub. Use when asked to publish, upload or release a model.
---

# Publish a model on Hugging Face

## Preconditions (stop and tell the user if any fails)
- Final results exist in `results/tables/` for this model, with all seeds.
- `uv run pytest -q` passes.
- The chosen checkpoint is the best on **validation**, and its test metrics are in the table.

## Repo
- Hub repo: `<hf-user>/<model-name>` (published: `punctuation-casing-en-7m`, `meeting-notes-en-21m`).
  Confirm the name with the user before creating it.
- Login via `HF_TOKEN` environment variable or `huggingface-cli login`. Never write the token to a file in the repo.

## Folder to upload (`release/<model-name>/`)
- `model.keras`: the final model
- `config.json`: architecture hyperparameters (vocab, d_model, layers, heads, FFN, max length, heads)
- `tokenizer.json` and `vocab.txt`: our WordPiece tokenizer
- `README.md`: the model card (below)
- `results.md`: the results table for this model
- `example.py`: load and run on one example
- `modeling.py`: self-contained layers + inference (needs only `keras`, `jax`, `tokenizers`)

The card, `modeling.py` and `example.py` are hand-written in `src/publish/hub/<model-name>/`;
`uv run python -m src.publish.package` copies them into the release folder and replaces their
`<hf-user>` placeholder with `$HF_NAMESPACE`. The repository never names the Hub account: keep
`<hf-user>` in every committed file. The custom layers
are registered under `package="dialogue_models"`, so `modeling.py` repeats them: keep their
`get_config` identical to `src/`, and check that it gives the same outputs as the repo code
before uploading (run both in separate processes: they register the same names).

## Model card (`README.md`)
YAML header: `library_name: keras`, `language: en`, `license`, `datasets`, `metrics`, `tags`
(`keras`, `jax`, `from-scratch`, plus task tags such as `meeting-notes` or `punctuation-restoration`).

Sections:
1. **Summary**: what it does, params, size on disk, CPU latency on M2.
2. **Built from scratch**: own tokenizer, own architecture, Keras 3 + JAX; for Model 1, MLM pretraining and what the teachers were used for (the released v3 checkpoint is gold-only).
3. **Architecture**: table from `config.json`.
4. **Training data**: datasets, splits, noise process (Model 2), teacher prompt and label validation (Model 1).
5. **Results**: table with baselines and teachers, mean ± std over seeds.
6. **Usage**: install line + short snippet from `example.py`, with a before/after example.
7. **Limitations**: English only; domain (WikiText, DialogSum); Model 2 does not fix misrecognised words; Model 1 is extractive (v3: gold labels; v1/v2: agreement with the teacher, not human labels).
8. **Licenses**: list the license of every dataset and teacher. DialogSum is non-commercial,
   so the model license must be compatible; check the dataset licenses and confirm the final license with the user before uploading.

## Upload
- `huggingface_hub.HfApi().create_repo(..., exist_ok=True)` then `upload_folder(...)`.

## Verify after upload
- In a fresh virtual environment with only `keras`, `jax`, `tokenizers` and `huggingface_hub`
  (no repo code), run the Hub's `example.py`: it downloads the files with
  `huggingface_hub.snapshot_download` and loads `model.keras` through `modeling.py`. Check the
  output matches the local run.
- Report the Hub URL and the verification result.
