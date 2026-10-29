# Dash demo of both models, CPU only: `docker compose up --build` (see docker-compose.yml).
FROM python:3.11-slim

# uv from its official image; dependencies come from uv.lock (same versions as development).
COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/srv/.venv \
    PATH="/srv/.venv/bin:$PATH" \
    KERAS_BACKEND=jax \
    HF_HOME=/cache/huggingface \
    PORT=8050

WORKDIR /srv
# Dependencies first (cached layer): runtime deps + the `app` extra, no dev tools.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra app --no-install-project

# Only the inference code: the models come from the Hugging Face Hub or a mounted folder.
COPY src/ src/
COPY app/ app/

RUN useradd --create-home --uid 1000 demo && mkdir -p /cache/huggingface \
    && chown -R demo /cache
USER demo

EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/', timeout=4)"

# gunicorn (production WSGI server) instead of Dash's development server. One worker, since
# each worker loads both models; threads serve concurrent requests. The first start downloads
# the models from the Hub, hence the long timeout.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 600 'app.main:create_server()'"]
