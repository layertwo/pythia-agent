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
COPY src/ ./src/

# Reproducible, production install: no dev deps, no editable mode
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.14-slim-trixie AS runtime

WORKDIR /app

# Only the built venv crosses over — no uv binary, no build tooling.
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY config.yaml ./

EXPOSE 8080

CMD ["python", "-m", "pythia_agent.server"]
