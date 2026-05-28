"""Fake Ollama HTTP server for integration tests.

Stubs the two Ollama endpoints mem0 and Strands hit:
- POST /api/chat       (LLM completions; branches for mem0 fact-extraction prompts)
- POST /api/embeddings (vector embeddings)

Returns deterministic outputs so integration tests are reproducible without
running a real Ollama instance or downloading any models.

EMBEDDING_DIMS (768) must match `embedder.embedding_dims` in config.yaml.
"""

import hashlib
import json
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

EMBEDDING_DIMS = 768

app = FastAPI(title="fake-ollama")


class EmbeddingRequest(BaseModel):
    model: str
    prompt: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False


def _seeded_vector(text: str, dims: int) -> list[float]:
    """Deterministic, normalised-ish vector seeded from the input text."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


def _looks_like_fact_extraction(messages: list[ChatMessage]) -> bool:
    """Heuristic: mem0's fact-extraction prompt mentions JSON and facts."""
    joined = " ".join(m.content.lower() for m in messages)
    return ("json" in joined and "fact" in joined) or "extract facts" in joined


@app.post("/api/embeddings")
def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
    return {"embedding": _seeded_vector(req.prompt, EMBEDDING_DIMS)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming not supported by fake-ollama")

    if _looks_like_fact_extraction(req.messages):
        # Echo the last user message as a single extracted fact.
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        content = json.dumps({"facts": [f"user said: {last_user}"]})
    else:
        content = "Acknowledged."

    return {
        "model": req.model,
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
