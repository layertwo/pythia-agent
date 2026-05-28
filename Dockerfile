FROM python:3.14-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install --system -e "."

COPY config.yaml ./

EXPOSE 8080

CMD ["python", "-m", "pythia_agent.server"]
