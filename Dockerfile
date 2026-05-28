FROM python:3.14-slim-bookworm

# Copy uv binary from the official image (faster than `pip install uv`)
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

WORKDIR /app

# Bytecompile installed packages and copy (not hardlink) into the image layer
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Reproducible, production install: no dev deps, no editable mode
RUN uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

COPY config.yaml ./

EXPOSE 8080

CMD ["python", "-m", "pythia_agent.server"]
