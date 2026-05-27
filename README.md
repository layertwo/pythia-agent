# pythia-agent

Self-hosted AI agent built on [Strands Agents SDK](https://strandsagents.com) with persistent memory via [mem0](https://mem0.ai).

## Features

- **Persistent Memory** — mem0-powered long-term memory with Qdrant vector store
- **Dual Memory Mode** — automatic context injection + explicit remember/recall/forget tools
- **Configurable Model Providers** — Ollama (default), OpenAI, Anthropic, AWS Bedrock, LiteLLM
- **Docker-first** — single `docker compose up` to run the full stack
- **Config file + env overrides** — `config.yaml` for defaults, environment variables for runtime

## Quick Start

```bash
# Start the full stack (pythia + qdrant)
docker compose up --build

# Test health
curl http://localhost:8080/health

# Chat with the agent
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello! My name is Lucas and I like Python.", "user_id": "lucas"}'

# Search memories
curl -X POST http://localhost:8080/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my name?", "user_id": "lucas"}'

# Get all memories for a user
curl http://localhost:8080/memory/lucas
```

## Configuration

Edit `config.yaml` to change defaults. Environment variables with `PYTHIA_` prefix override config values.

### Switch Model Provider

```bash
# Use OpenAI
PYTHIA_MODEL_PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up --build

# Use Anthropic
PYTHIA_MODEL_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

### Ollama Setup

The default config expects Ollama running on the host. From inside Docker, it connects via `host.docker.internal:11434`.

```bash
# Pull required models on host
ollama pull llama3.1
ollama pull nomic-embed-text
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message, get a response |
| POST | `/memory/search` | Search memories for a user |
| GET | `/memory/{user_id}` | Get all memories for a user |
| DELETE | `/memory/{memory_id}` | Delete a specific memory |
| GET | `/health` | Health check with config info |

## Architecture

The memory system follows the same pattern as [AgentCore Memory SessionManager](https://strandsagents.com/docs/community/session-managers/agentcore-memory/) — implementing the Strands `SessionManager` interface to hook into the agent lifecycle:

- **`MessageAddedEvent` hook** — retrieves relevant memories and injects them as a `<memory_context>` block in the user message before the model sees it
- **`AfterInvocationEvent` hook** — stores the conversation exchange as memories after each turn
- **Explicit tools** — `remember`, `recall`, `forget`, `list_memories` are class-based `@tool` methods the model can call directly

## Plugins

Tools are packaged as Strands `Plugin` classes with auto-discovered `@tool` methods:

| Plugin | Tools |
|--------|-------|
| `SystemToolsPlugin` | `current_time`, `shell`, `python_exec`, `file_read`, `file_write`, `calculator` |
| `WebToolsPlugin` | `http_request`, `exa_search`, `tavily_search`, `rss_read` |
| `AgentToolsPlugin` | `think`, `use_llm`, `stop`, `journal_write`, `journal_read`, `journal_list` |
| `Mem0SessionManager` | `remember`, `recall`, `forget`, `list_memories` (via tools list, not plugin) |

Web search tools require API keys: `EXA_API_KEY`, `TAVILY_API_KEY`.

## Project Structure

```
pythia-agent/
├── src/pythia_agent/
│   ├── agent.py          # PythiaAgent - creates Agent with plugins
│   ├── config.py         # Settings - pydantic config (yaml + env)
│   ├── memory.py         # Mem0SessionManager - SessionManager + tools
│   ├── server.py         # PythiaServer - FastAPI with per-user agents
│   ├── plugins/
│   │   ├── system_tools.py  # Shell, files, python, time, calc
│   │   ├── web_tools.py     # HTTP, Exa, Tavily, RSS
│   │   └── agent_tools.py   # Think, sub-agent, stop, journal
│   └── providers/
│       └── factory.py    # ModelFactory - model provider creation
├── config.yaml           # Default configuration
├── docker-compose.yaml   # Full stack: pythia + qdrant
├── Dockerfile
└── pyproject.toml
```

## Development

```bash
# Install locally
pip install -e ".[dev]"

# Run without Docker (requires local Qdrant + Ollama)
python -m pythia_agent.server
```
