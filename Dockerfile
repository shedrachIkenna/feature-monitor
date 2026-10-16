# syntax=docker/dockerfile:1
# Feature Monitor on CPU, in three stages; only the last one ships.
#   builder: installs the locked Python packages into /app/.venv with uv.
#   assets:  downloads GPT-2, the SAE, the embedder and the baseline features' labels.
#   runtime: a fresh slim image with the venv, the downloads and the code, run as uid 1000.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/
# Compile .pyc files now so startup is faster; copy (not hard-link) out of the uv cache mount;
# use the image's Python instead of downloading one.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-dev --no-install-project

FROM builder AS assets
COPY backend/ backend/
COPY monitor/ monitor/
# Download into a cache mount that outlives the build, so a retry after a failed download (or a
# rebuild after a code change) reuses what is already there, then copy it into the image.
RUN --mount=type=cache,target=/hf-cache \
    HF_HOME=/hf-cache .venv/bin/python -m monitor.prefetch models \
    && mkdir -p /app/.cache && cp -a /hf-cache /app/.cache/huggingface
RUN .venv/bin/python -m monitor.prefetch labels

FROM ${PYTHON_IMAGE} AS runtime
# Hugging Face Spaces runs containers as uid 1000, so that user owns everything it writes to.
RUN useradd --create-home --uid 1000 user && mkdir /data && chown user:user /data
# HF_HUB_OFFLINE: load models from HF_HOME only, never from the network.
# REQUIRE_RECORD_TOKEN: refuse recorded prompts unless FEATURE_MONITOR_RECORD_TOKEN is set.
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    FEATURE_MONITOR_DB=/data/feature_monitor.db \
    FEATURE_MONITOR_REQUIRE_RECORD_TOKEN=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder --chown=user:user /app/.venv .venv
COPY --from=assets --chown=user:user /app/.cache .cache
COPY --chown=user:user backend/ backend/
COPY --chown=user:user frontend/ frontend/
COPY --chown=user:user monitor/ monitor/
# Writable: the app adds labels of features outside the baseline on first use.
COPY --from=assets --chown=user:user /app/monitor/labels_cache.json monitor/labels_cache.json
USER user
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
