import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pythia_agent.agent import PythiaAgent
from pythia_agent.environment import ServiceProvider
from pythia_agent.utils import utc_now

logger = logging.getLogger(__name__)

MAX_CACHED_AGENTS = 50

_provider = ServiceProvider()


class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    user_id: str
    model_provider: str
    timestamp: str


class MemorySearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    top_k: int = 5


_agents: OrderedDict[str, PythiaAgent] = OrderedDict()


def _get_agent(user_id: str) -> PythiaAgent:
    if user_id in _agents:
        _agents.move_to_end(user_id)
        return _agents[user_id]

    agent = _provider.create_agent(user_id)
    _agents[user_id] = agent

    while len(_agents) > MAX_CACHED_AGENTS:
        _agents.popitem(last=False)

    return agent


def _get_session_manager(user_id: str):
    agent = _get_agent(user_id)
    if not agent.session_manager:
        raise HTTPException(status_code=503, detail="Memory system is disabled")
    return agent.session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    _provider.start()
    logger.info("Service provider started")
    try:
        yield
    finally:
        _provider.shutdown()
        logger.info("Service provider shut down")


app = FastAPI(title=_provider.settings.agent.name, version="0.1.0", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        agent = _get_agent(request.user_id)
        result = agent.invoke(request.prompt)
        return ChatResponse(
            response=result["response"],
            user_id=request.user_id,
            model_provider=_provider.settings.model.provider,
            timestamp=utc_now().isoformat(),
        )
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/memory/search")
async def search_memories(request: MemorySearchRequest) -> dict[str, Any]:
    sm = _get_session_manager(request.user_id)
    memories = sm.search(request.query, top_k=request.top_k)
    return {"results": memories, "user_id": request.user_id}


@app.get("/memory/{user_id}")
async def get_memories(user_id: str) -> dict[str, Any]:
    sm = _get_session_manager(user_id)
    memories = sm.get_all()
    return {"results": memories, "user_id": user_id}


@app.get("/memory/{user_id}/{memory_id}")
async def get_memory(user_id: str, memory_id: str) -> dict[str, Any]:
    sm = _get_session_manager(user_id)
    memory = sm.get(memory_id)
    return {"result": memory, "user_id": user_id}


@app.delete("/memory/{user_id}/{memory_id}")
async def delete_memory(user_id: str, memory_id: str) -> dict[str, str]:
    sm = _get_session_manager(user_id)
    sm.delete(memory_id)
    return {"status": "deleted", "memory_id": memory_id}


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = _provider.settings
    return {
        "status": "healthy",
        "agent": settings.agent.name,
        "model_provider": settings.model.provider,
        "model_id": settings.model.model_id,
        "memory_enabled": settings.memory.enabled,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "pythia_agent.server:app",
        host=_provider.settings.server.host,
        port=_provider.settings.server.port,
        reload=False,
        ws="none",
    )


if __name__ == "__main__":
    main()
