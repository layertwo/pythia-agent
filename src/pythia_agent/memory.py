"""Mem0-based session manager for Pythia agent.

Follows the same architectural pattern as AgentCoreMemorySessionManager:
- Extends SessionManager to integrate with the Strands agent lifecycle
- Uses hooks to inject retrieved memories into user messages before model calls
- Stores conversation exchanges as memories after each invocation
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from mem0 import Memory
from strands import tool
from strands.hooks import AfterInvocationEvent, MessageAddedEvent
from strands.hooks.registry import HookRegistry
from strands.session.session_manager import SessionManager
from strands.types.content import Message

from pythia_agent.config import MemoryConfig

if TYPE_CHECKING:
    from strands.agent.agent import Agent

logger = logging.getLogger(__name__)

CONTEXT_TAG = "memory_context"


class Mem0SessionManager(SessionManager):
    """Mem0-based session manager providing persistent memory for Strands agents.

    Integrates with the Strands agent lifecycle to:
    - Retrieve relevant memories and inject them into user messages (via MessageAddedEvent hook)
    - Store conversation exchanges as memories after each invocation (via AfterInvocationEvent hook)
    - Expose explicit memory tools (remember/recall/forget/list_memories) for agent use
    """

    # Single-worker executor shared across all session managers so writes
    # serialize globally. mem0's Memory.add touches an SQLite history DB and
    # has no internal locking; serializing avoids "database is locked" and
    # gives us a clean drain point on shutdown.
    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mem0-write"
    )

    def __init__(self, config: MemoryConfig, user_id: str = "default"):
        self.config = config
        self.user_id = user_id
        self._client: Optional[Memory] = None
        self._client_lock = threading.Lock()

    @property
    def client(self) -> Memory:
        # Double-checked locking: the background-write thread and the
        # foreground asyncio search/auto-inject path can race on first access.
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = Memory.from_config(self._build_mem0_config())
                    logger.info(
                        "mem0 client initialized (llm=%s, embedder=%s, vector_store=%s)",
                        self.config.llm.provider,
                        self.config.embedder.provider,
                        self.config.vector_store.provider,
                    )
        return self._client

    @classmethod
    def shutdown_writes(cls) -> None:
        """Drain pending background memory writes. Called from app shutdown."""
        cls._executor.shutdown(wait=True)

    def _build_mem0_config(self) -> dict[str, Any]:
        cfg = self.config
        config: dict[str, Any] = {
            "llm": {
                "provider": cfg.llm.provider,
                "config": {
                    "model": cfg.llm.model,
                    "temperature": cfg.llm.temperature,
                    "max_tokens": cfg.llm.max_tokens,
                },
            },
            "embedder": {
                "provider": cfg.embedder.provider,
                "config": {
                    "model": cfg.embedder.model,
                    "embedding_dims": cfg.embedder.embedding_dims,
                },
            },
            "vector_store": {
                "provider": cfg.vector_store.provider,
                "config": {
                    "collection_name": cfg.vector_store.collection_name,
                    "host": cfg.vector_store.host,
                    "port": cfg.vector_store.port,
                    "user": cfg.vector_store.user,
                    "password": cfg.vector_store.password,
                    "dbname": cfg.vector_store.dbname,
                    # Must match embedder.embedding_dims; mem0's pgvector config defaults to 1536.
                    "embedding_model_dims": cfg.embedder.embedding_dims,
                },
            },
        }

        if cfg.llm.provider == "ollama":
            config["llm"]["config"]["ollama_base_url"] = cfg.llm.ollama_base_url
        if cfg.embedder.provider == "ollama":
            config["embedder"]["config"]["ollama_base_url"] = cfg.embedder.ollama_base_url

        return config

    # ------------------------------------------------------------------
    # SessionManager interface
    # ------------------------------------------------------------------

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for memory retrieval and storage."""
        if self.config.auto_inject:
            registry.add_callback(MessageAddedEvent, self._on_message_added)
        if self.config.auto_store:
            registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize the session manager with an agent."""
        logger.info("Mem0SessionManager initialized for user_id=%s", self.user_id)

    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Append a message to the agent's conversation. No-op for mem0 (stateless per-turn)."""
        pass

    def redact_latest_message(self, redact_message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Redact the latest message. No-op for mem0."""
        pass

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Sync agent state. No-op for mem0 (memories are stored via hooks)."""
        pass

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Retrieve relevant memories and inject into the user's message."""
        messages = event.agent.messages
        if not messages or messages[-1].get("role") != "user":
            return

        content = messages[-1].get("content", [])
        if not content or "text" not in content[-1]:
            return

        user_query = content[-1]["text"]
        memories = self.search(
            user_query,
            top_k=self.config.auto_inject_top_k,
            min_score=self.config.auto_inject_min_score,
        )

        if not memories:
            return

        context_lines = []
        for mem in memories:
            context_lines.append(mem.get("memory", ""))

        context_text = "\n".join(context_lines)
        messages[-1]["content"].insert(
            0, {"text": f"<{CONTEXT_TAG}>{context_text}</{CONTEXT_TAG}>"}
        )
        logger.info("Injected %d memories into user message", len(memories))

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Store the conversation exchange as memories after invocation completes."""
        messages = event.agent.messages
        if len(messages) < 2:
            return

        conversation = []
        for msg in messages[-2:]:
            role = msg.get("role")
            content_blocks = msg.get("content", [])
            for block in content_blocks:
                if "text" in block:
                    text = block["text"]
                    if not text.startswith(f"<{CONTEXT_TAG}>"):
                        conversation.append({"role": role, "content": text})
                    break

        if conversation:
            self._executor.submit(self._store_in_background, conversation)

    def _store_in_background(self, conversation: list[dict]) -> None:
        try:
            self.add(conversation)
            logger.debug("Stored %d messages as memories", len(conversation))
        except Exception:
            logger.exception("Background memory store failed")

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        k = top_k or self.config.auto_inject_top_k
        kwargs: dict[str, Any] = {
            "query": query,
            "filters": {"user_id": self.user_id},
            "top_k": k,
        }
        if min_score is not None:
            kwargs["threshold"] = min_score
        results = self.client.search(**kwargs)
        return results.get("results", [])

    def add(self, messages: list[dict], metadata: dict | None = None) -> dict:
        return self.client.add(messages, user_id=self.user_id, metadata=metadata)

    def get_all(self) -> list[dict]:
        results = self.client.get_all(filters={"user_id": self.user_id})
        return results.get("results", [])

    def get(self, memory_id: str) -> dict:
        return self.client.get(memory_id)

    def delete(self, memory_id: str) -> None:
        self.client.delete(memory_id)

    def history(self, memory_id: str) -> list[dict]:
        return self.client.history(memory_id)

    # ------------------------------------------------------------------
    # Strands tools (class-based, passed to Agent via get_tools())
    # ------------------------------------------------------------------

    @tool
    def remember(self, content: str, metadata: str = "") -> str:
        """Store important information in long-term memory.

        Use this to explicitly save facts, preferences, or context the user
        wants remembered across conversations.

        Args:
            content: The fact, preference, or information to store
            metadata: Optional JSON metadata string to attach to the memory
        """
        meta = None
        if metadata:
            try:
                meta = json.loads(metadata)
            except json.JSONDecodeError:
                meta = {"note": metadata}

        messages = [{"role": "user", "content": content}]
        result = self.add(messages, metadata=meta)
        stored = result.get("results", [])
        return f"Stored {len(stored)} memory item(s)."

    @tool
    def recall(self, query: str) -> str:
        """Search long-term memory for relevant information.

        Use this to retrieve stored facts, preferences, or past context
        that may be relevant to the current conversation.

        Args:
            query: What to search for in memory
        """
        memories = self.search(query)
        if not memories:
            return "No relevant memories found."

        lines = [f"Found {len(memories)} relevant memories:"]
        for mem in memories:
            score = mem.get("score", 0)
            lines.append(f"- [{score:.2f}] {mem.get('memory', '')} (id: {mem.get('id', 'unknown')})")
        return "\n".join(lines)

    @tool
    def forget(self, memory_id: str) -> str:
        """Delete a specific memory by its ID.

        Use this when the user explicitly asks to forget or remove something.

        Args:
            memory_id: The ID of the memory to delete
        """
        self.delete(memory_id)
        return f"Memory '{memory_id}' has been deleted."

    @tool
    def list_memories(self) -> str:
        """List all stored memories for the current user.

        Use this when the user asks what you remember about them.
        """
        memories = self.get_all()
        if not memories:
            return "No memories stored."

        lines = [f"All memories ({len(memories)} total):"]
        for mem in memories:
            lines.append(f"- {mem.get('memory', '')} (id: {mem.get('id', 'unknown')})")
        return "\n".join(lines)

    def get_tools(self) -> list:
        """Return bound tool methods for the Strands agent."""
        return [self.remember, self.recall, self.forget, self.list_memories]
