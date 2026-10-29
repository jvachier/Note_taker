# Security policy

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes |
| Published models (`punctuation-casing-en-7m`, `meeting-notes-en-21m`), latest revision | Yes |
| Anything older | No |

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately through GitHub: on this repository's page, open
**Security → Advisories → Report a vulnerability**.

Please include:
- what you found and where (file, commit, model repo and revision);
- how to reproduce it;
- the impact you expect.

You should receive an acknowledgement within 7 days. Confirmed issues are fixed on `main` and,
where they affect a published model, in a new revision of its Hugging Face repository.

## Scope

In scope:
- **Demo app** (`app/`, `Dockerfile`, `docker-compose.yml`): for example, input that crashes
  the server or blocks it for a long time despite the input limit, cross-site scripting through
  rendered text, or a container that runs with more privileges than needed.
- **Model loading** (`app/models.py`, the Hub `modeling.py` files): models are loaded with
  `keras.saving.load_model` in its default safe mode. Report any way a model folder or Hub
  revision could run code that was not shipped in the published `modeling.py`.
- **Pipeline code**: unsafe handling of downloaded data (datasets, Kaggle outputs), path
  traversal when copying files, or vulnerable pinned dependencies (`uv.lock`).
- **CI and pre-commit hooks** (`.github/`, `.pre-commit-config.yaml`).

Out of scope:
- Wrong or poor model outputs (missed decisions, bad punctuation): please open a normal issue.
- Vulnerabilities in upstream projects (Keras, JAX, Dash, the datasets): report those to their
  maintainers.
- Personal information in the upstream datasets (DialogSum, WikiText, AMI): report it to their
  maintainers; we will remove affected examples from this repository if any are included.

## Secrets

The code needs no secrets to train or run. Tokens are used only in manual steps: `HF_TOKEN` to
publish models (or to read private Hub repos in the app) and Kaggle credentials in
`~/.kaggle/kaggle.json` to fetch notebook outputs. They must never be committed or written into
the repository. If you find a token in the repository or its history, report it privately as
described above.
