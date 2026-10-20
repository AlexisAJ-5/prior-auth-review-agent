# syntax=docker/dockerfile:1
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    EMBEDDING_CACHE_DIR=/models \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first, so code edits reuse the cached layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the embedding model (~70 MB) into the image: no model download at runtime.
ARG PRELOAD_EMBEDDINGS=1
RUN if [ "$PRELOAD_EMBEDDINGS" = "1" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models')"; \
    fi

COPY data ./data
COPY evals/cases ./evals/cases

RUN useradd --create-home --uid 10001 app && mkdir -p /app/storage /models && chown -R app /app/storage /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "prior_auth.api:app", "--host", "0.0.0.0", "--port", "8000"]
