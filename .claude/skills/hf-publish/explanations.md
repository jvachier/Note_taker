# `hf-publish`: what this skill does

A [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills): instructions Claude
loads when you ask it to publish a model. Claude follows [`SKILL.md`](SKILL.md); this file
explains it for people.

## Purpose

Package one trained model and publish it on the Hugging Face Hub so that anyone can download
and run it **without this repository**: only `keras`, `jax`, `tokenizers` and
`huggingface_hub` are needed.

Published with it so far:
`<hf-user>/punctuation-casing-en-7m`
(Model 2) and
`<hf-user>/meeting-notes-en-21m`
(Model 1 v3).

## When it runs

When you ask Claude to "publish", "upload" or "release" a model, or type `/hf-publish`.

## What it does

1. **Checks preconditions** and stops if one fails: all seeds are in `results/tables/`, the
   tests pass, and the released checkpoint is the best one on **validation** (never test).
2. **Packages** the model with `uv run python -m src.publish.package`. Each folder
   `release/<repo>/` gets:
   - `model.keras` (no optimizer state), `tokenizer.json`, `vocab.txt`;
   - `config.json`: the architecture, tuned thresholds, inference rules, and the run and seed
     the weights came from;
   - `results.md`: the model's results table;
   - from `src/publish/hub/<repo>/`: the model card `README.md`, `modeling.py` (standalone
     layers and inference) and `example.py`.
3. **Checks `modeling.py` against the repo code**: same outputs on the same inputs, run in
   separate processes because both register the same layer names.
4. **Asks you** to confirm the repo name and the licence. DialogSum is non-commercial, so the
   models are CC BY-NC-SA 4.0.
5. **Uploads** with `HfApi().create_repo(...)` and `upload_folder(...)`.
6. **Verifies** in a fresh virtual environment with no repo code: it runs the Hub's
   `example.py` and checks that the output matches the local run.

## Guardrails

- Never writes the Hugging Face token to a file; it uses `HF_TOKEN` or `hf auth login`.
- Never publishes teacher or baseline weights (Qwen, RoBERTa, DistilBERT).
- The model card must list limitations and the licence of every dataset.

## Example request

> "Publish the meeting-notes model on Hugging Face."

## Related

- [`run-experiment`](../run-experiment/explanations.md) produces the results tables this skill
  needs.
- [`kaggle-stage`](../kaggle-stage/explanations.md) trains the checkpoints.
