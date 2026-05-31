"""Fake Ollama HTTP server for integration tests.

Stubs the Ollama endpoints mem0 and Strands hit:
- GET  /api/tags       (model list; mem0 embedder calls this to verify the model exists)
- POST /api/chat       (LLM completions; streaming for Strands, non-streaming for mem0)
- POST /api/embed      (vector embeddings; newer endpoint used by mem0 via ollama client)
- POST /api/embeddings (legacy vector embeddings endpoint)

Returns deterministic outputs so integration tests are reproducible without
running a real Ollama instance or downloading any models.

EMBEDDING_DIMS (768) must match `embedder.embedding_dims` in config.yaml.
"""

import hashlib
import json
import random
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

EMBEDDING_DIMS = 768
# Models advertised to any caller that checks /api/tags before embedding/chatting.
FAKE_MODELS = ["nomic-embed-text", "nomic-embed-text-v2-moe", "llama3.1"]

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


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]


def _seeded_vector(text: str, dims: int) -> list[float]:
    """Deterministic, normalised-ish vector seeded from the input text."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


def _looks_like_fact_extraction(messages: list[ChatMessage]) -> bool:
    """Heuristic: detect mem0's memory-extraction prompts.

    mem0 2.0.4 uses the ADDITIVE_EXTRACTION_PROMPT which mentions "Memory Extractor"
    and "## New Messages" in the system/user prompts. Older versions used "facts".
    We check for both styles so the stub stays forward-compatible.
    """
    joined = " ".join(m.content.lower() for m in messages)
    # mem0 2.0.4 ADDITIVE_EXTRACTION_PROMPT style
    if "memory extractor" in joined or "## new messages" in joined:
        return True
    # Older mem0 FACT_RETRIEVAL_PROMPT style (kept for backwards-compat)
    if ("json" in joined and "fact" in joined) or "extract facts" in joined:
        return True
    return False


@app.get("/api/tags")
def tags() -> dict[str, Any]:
    """Return a model list so callers don't attempt to pull models."""
    return {
        "models": [
            {"name": f"{m}:latest", "model": f"{m}:latest"}
            for m in FAKE_MODELS
        ]
    }


@app.post("/api/embeddings")
def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
    return {"embedding": _seeded_vector(req.prompt, EMBEDDING_DIMS)}


@app.post("/api/embed")
def embed(req: EmbedRequest) -> dict[str, Any]:
    """Newer /api/embed endpoint (used by ollama client's embed() method).

    Accepts a single string or list of strings; returns embeddings as a list of vectors.
    """
    inputs = req.input if isinstance(req.input, list) else [req.input]
    return {"embeddings": [_seeded_vector(text, EMBEDDING_DIMS) for text in inputs]}


def _chat_content(req: ChatRequest) -> str:
    """Return the assistant content for a chat request.

    For mem0 fact-extraction calls (mem0 2.0.4 ADDITIVE_EXTRACTION_PROMPT style):
      - Returns ``{"memory": [{"id": "0", "text": "...", "event": "NONE"}]}``
      - mem0 expects each item to have at minimum ``text`` and ``event`` fields.
    For regular chat calls: returns "Acknowledged."
    """
    if _looks_like_fact_extraction(req.messages):
        # Extract a short summary from the last user message.
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        # Use just the first 100 chars to keep the fact concise.
        fact_text = (last_user[:100] + "...") if len(last_user) > 100 else last_user
        return json.dumps({
            "memory": [{"id": "0", "text": fact_text, "event": "NONE"}]
        })
    return "Acknowledged."


async def _stream_chat(model: str, content: str) -> AsyncGenerator[bytes, None]:
    """Yield NDJSON lines for a streaming chat response.

    The final chunk must include eval_count, prompt_eval_count, and total_duration
    so the strands OllamaModel can compute token usage metadata without errors.
    """
    # Send the full content in a single chunk then close.
    chunk = {
        "model": model,
        "message": {"role": "assistant", "content": content, "tool_calls": None},
        "done": False,
    }
    yield (json.dumps(chunk) + "\n").encode()
    done_chunk = {
        "model": model,
        "message": {"role": "assistant", "content": "", "tool_calls": None},
        "done": True,
        "done_reason": "stop",
        # Fake token counts required by strands' metadata format_chunk handler.
        "prompt_eval_count": 10,
        "eval_count": 5,
        "total_duration": 100000000,  # 100ms in nanoseconds
    }
    yield (json.dumps(done_chunk) + "\n").encode()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> Any:
    content = _chat_content(req)

    if req.stream:
        return StreamingResponse(
            _stream_chat(req.model, content),
            media_type="application/x-ndjson",
        )

    return {
        "model": req.model,
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
