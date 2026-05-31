FROM python:3.14-slim-trixie AS builder

# Copy uv binary from the official image (faster than `pip install uv`)
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

WORKDIR /app

# Bytecompile installed packages, copy (not hardlink) into the layer, and skip
# the download/build cache so it doesn't bloat the venv we carry forward.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md ./

# Install third-party deps first against just the lockfile so this layer
# stays cached across pure src/ changes. --no-install-project skips
# building our own package (which needs src/ to be present).
RUN uv sync --frozen --no-dev --no-editable --no-install-project

COPY src/ ./src/

# Final sync builds + installs the project itself. Cheap because every
# dependency is already present from the layer above.
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.14-slim-trixie AS runtime

# OCI labels — picked up on per-arch images. GHCR's package page reads
# org.opencontainers.image.description from the multi-arch index, set in
# the merge job via `docker buildx imagetools create --annotation`.
LABEL org.opencontainers.image.source="https://github.com/layertwo/pythia-agent"
LABEL org.opencontainers.image.description="Self-hosted AI agent built on Strands Agents SDK with persistent memory via mem0 and PostgreSQL (pgvector)."
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Only the built venv crosses over — no uv binary, no build tooling.
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY config.yaml ./

EXPOSE 8080

CMD ["python", "-m", "pythia_agent.server"]
