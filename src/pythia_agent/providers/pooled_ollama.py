"""OllamaModel subclass that reuses an AsyncClient per asyncio loop.

Strands' stock OllamaModel constructs `ollama.AsyncClient(...)` on every
`stream` / `structured_output` call, discarding the httpx connection pool
and forcing a fresh TLS handshake per turn against https://ollama.com.

httpx.AsyncClient binds its pool to the loop that first uses it, so a
single cached client breaks when Strands' sync `Agent.__call__` routes
through `run_async()` (a fresh `asyncio.run()` per call, in its own
thread) — second use raises "Event loop is closed". The fix is to cache
one client per loop and rebuild when the loop is gone.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

import ollama
from pydantic import BaseModel
from strands.models.ollama import OllamaModel
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import override

T = TypeVar("T", bound=BaseModel)


class PooledOllamaModel(OllamaModel):
    """Caches one `ollama.AsyncClient` per running event loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._clients: dict[asyncio.AbstractEventLoop, ollama.AsyncClient] = {}

    def _get_client(self) -> ollama.AsyncClient:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None or loop.is_closed():
            client = ollama.AsyncClient(self.host, **self.client_args)
            self._clients[loop] = client
        return client

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        # Mirror upstream's stream() body, but use the cached client.
        # Upstream lives at strands.models.ollama.OllamaModel.stream.
        from strands.models._validation import warn_on_tool_choice_not_supported

        warn_on_tool_choice_not_supported(tool_choice)
        request = self.format_request(messages, tool_specs, system_prompt)
        tool_requested = False

        response = await self._get_client().chat(**request)

        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        event = None
        async for event in response:
            for tool_call in event.message.tool_calls or []:
                yield self.format_chunk(
                    {
                        "chunk_type": "content_start",
                        "data_type": "tool",
                        "data": tool_call,
                    }
                )
                yield self.format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "tool",
                        "data": tool_call,
                    }
                )
                yield self.format_chunk(
                    {
                        "chunk_type": "content_stop",
                        "data_type": "tool",
                        "data": tool_call,
                    }
                )
                tool_requested = True

            yield self.format_chunk(
                {
                    "chunk_type": "content_delta",
                    "data_type": "text",
                    "data": event.message.content,
                }
            )

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        # Guard the empty-stream case (e.g. cloud rate-limited mid-handshake):
        # `event` is None when the response yielded zero chunks; close the
        # message cleanly instead of dereferencing None.
        if event is None:
            yield self.format_chunk({"chunk_type": "message_stop", "data": "end_turn"})
            return
        yield self.format_chunk(
            {
                "chunk_type": "message_stop",
                "data": "tool_use" if tool_requested else event.done_reason,
            }
        )
        yield self.format_chunk({"chunk_type": "metadata", "data": event})

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        formatted_request = self.format_request(messages=prompt, system_prompt=system_prompt)
        formatted_request["format"] = output_model.model_json_schema()
        formatted_request["stream"] = False

        response = await self._get_client().chat(**formatted_request)

        try:
            content = response.message.content.strip()
            yield {"output": output_model.model_validate_json(content)}
        except Exception as e:
            raise ValueError(f"Failed to parse or load content into model: {e}") from e
