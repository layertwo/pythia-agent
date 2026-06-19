import logging
from collections.abc import AsyncIterator

from strands import Agent

from pythia_agent.memory import Mem0SessionManager

logger = logging.getLogger(__name__)


class PythiaAgent:
    """Strands agent assembled from injected dependencies.

    Does not know about ServiceProvider — receives a model, plugins, and
    optional session manager directly.
    """

    def __init__(
        self,
        model,
        system_prompt: str,
        plugins: list,
        session_manager: Mem0SessionManager | None = None,
        conversation_manager=None,
        limits: dict | None = None,
    ):
        self.session_manager = session_manager
        # Per-invocation hard backstop forwarded to the Strands agent loop
        # (turns / output_tokens / total_tokens). None -> never pass the kwarg.
        self._limits = limits
        tools = session_manager.get_tools() if session_manager else []

        # conversation_manager=None is Strands' own default, so pass it straight
        # through — no need to conditionally omit the kwarg.
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            plugins=plugins,
            session_manager=session_manager,
            conversation_manager=conversation_manager,
        )

    def invoke(self, message: str) -> dict:
        result = self.agent(message, **self._invocation_kwargs())
        msg = result.message or {}
        text = "".join(
            block.get("text", "") for block in msg.get("content", []) if isinstance(block, dict) and "text" in block
        )
        return {"response": text}

    async def stream(self, message: str) -> AsyncIterator[str]:
        """Yield text chunks from the model as they're generated."""
        async for event in self.agent.stream_async(message, **self._invocation_kwargs()):
            data = event.get("data") if isinstance(event, dict) else None
            if isinstance(data, str) and data:
                yield data

    def _invocation_kwargs(self) -> dict:
        """Keyword args common to invoke() and stream() — only `limits` when set."""
        return {"limits": self._limits} if self._limits else {}
