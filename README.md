# pythia-agent

Self-hosted AI agent built on [Strands Agents SDK](https://strandsagents.com) with persistent memory via [mem0](https://mem0.ai) and PostgreSQL (pgvector).

## Features

- **Persistent Memory** — mem0 with pgvector for long-term recall across sessions
- **11 Plugins** — system tools, web search, scheduling, goals, tasks, personas, notifications, context compression, safety guardrails, session persistence, agent reasoning
- **Configurable Model Providers** — Ollama (default), OpenAI, Anthropic, AWS Bedrock, LiteLLM
- **Cron Scheduler** — autonomous recurring jobs that run without user interaction
- **Docker-first** — single `docker compose up` to run the full stack (pythia + postgres)
- **Config file + env overrides** — `config.yaml` for defaults, environment variables for runtime
- **Service Provider** — dependency injection via `environment/service_provider.py`

## Quick Start

```bash
# Ensure Ollama is running on host with required models
ollama pull llama3.1
ollama pull nomic-embed-text

# Start the full stack (pythia + postgres)
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
ollama pull llama3.1
ollama pull nomic-embed-text
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message, get a response |
| POST | `/memory/search` | Search memories for a user |
| GET | `/memory/{user_id}` | Get all memories for a user |
| GET | `/memory/{user_id}/{memory_id}` | Get a specific memory |
| DELETE | `/memory/{user_id}/{memory_id}` | Delete a specific memory |
| GET | `/health` | Health check with config info |

## Architecture

Memory follows the [AgentCore Memory SessionManager](https://strandsagents.com/docs/community/session-managers/agentcore-memory/) pattern — implementing the Strands `SessionManager` interface:

- **`MessageAddedEvent` hook** — retrieves relevant memories and injects them as a `<memory_context>` block before the model sees the message
- **`AfterInvocationEvent` hook** — stores the conversation exchange as memories after each turn
- **Explicit tools** — `remember`, `recall`, `forget`, `list_memories`

Plugins extend `strands.plugins.Plugin` with auto-discovered `@tool` and `@hook` methods. Each plugin contributes behavioral guidance to the system prompt via `init_agent`.

## Plugins

| Plugin | Tools / Hooks |
|--------|---------------|
| `SystemToolsPlugin` | `current_time`, `shell`, `python_exec`, `file_read`, `file_write`, `calculator` |
| `WebToolsPlugin` | `http_request`, `exa_search`, `tavily_search`, `rss_read` |
| `AgentToolsPlugin` | `think`, `use_llm`, `stop`, `journal_write`, `journal_read`, `journal_list` |
| `SchedulerPlugin` | `create_job`, `list_jobs`, `delete_job`, `toggle_job`, `job_history` + background cron thread |
| `GoalsPlugin` | `create_goal`, `update_goal`, `check_goals`, `delete_goal` |
| `TasksPlugin` | `update_tasks`, `get_tasks`, `clear_tasks` (in-memory, session-scoped) |
| `PersonasPlugin` | `create_persona`, `list_personas`, `switch_persona`, `get_persona`, `delete_persona` |
| `NotificationPlugin` | `notify_telegram`, `notify_webhook` |
| `SafetyPlugin` | Hooks: iteration budget (max 50 tool calls), repetitive call detection |
| `SessionsPlugin` | Hooks: conversation persistence. Tools: `search_sessions`, `list_sessions`, `resume_session` |
| `ContextPlugin` | Hook: auto-compresses context when nearing token limit. Tool: `get_context_stats` |
| `Mem0SessionManager` | `remember`, `recall`, `forget`, `list_memories` (via session manager, not plugin) |

Web search tools require API keys: `EXA_API_KEY`, `TAVILY_API_KEY`.

## Project Structure

```
pythia-agent/
├── src/pythia_agent/
│   ├── agent.py              # PythiaAgent - thin wrapper, receives injected deps
│   ├── config.py             # Pydantic settings (yaml + env)
│   ├── db.py                 # SQLAlchemy models + engine (Job, JobRun, Goal, Persona)
│   ├── memory.py             # Mem0SessionManager - SessionManager + memory tools
│   ├── server.py             # FastAPI server
│   ├── utils.py              # Shared utilities (slugify, utc_now, etc.)
│   ├── environment/
│   │   └── service_provider.py  # Composition root, dependency injection
│   ├── models/
│   │   └── session.py        # ConversationSession + SessionMessage models
│   ├── plugins/
│   │   ├── agent_tools.py    # Think, sub-agent, stop, journal
│   │   ├── context.py        # Context window compression
│   │   ├── goals.py          # Goal tracking
│   │   ├── notifications.py  # Telegram + webhook alerts
│   │   ├── personas.py       # Multi-personality management
│   │   ├── safety.py         # Iteration budget + loop detection
│   │   ├── scheduler.py      # Cron-based job scheduling
│   │   ├── sessions.py       # Conversation persistence + search
│   │   ├── system_tools.py   # Shell, files, python, time, calc
│   │   ├── tasks.py          # In-session task decomposition
│   │   └── web_tools.py      # HTTP, Exa, Tavily, RSS
│   └── providers/
│       └── factory.py        # Model provider factory
├── tests/                    # pytest suite with coverage
├── config.yaml               # Default configuration
├── docker-compose.yaml       # Full stack: pythia + postgres (pgvector)
├── Dockerfile
└── pyproject.toml
```

## Development

```bash
# Install locally
pip install -e ".[dev]"

# Run tests with coverage
pytest

# Run without Docker (requires local Postgres + Ollama)
python -m pythia_agent.server
```
