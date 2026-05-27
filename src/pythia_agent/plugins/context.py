"""Context plugin: manages context window compression for long conversations.

When conversation approaches the model's token limit, compresses middle turns
into a structured summary while protecting the first N (system prompt + initial
context) and last N (recent work) messages.
"""

import json
import logging
from typing import Any

from strands import Agent, tool
from strands.hooks import BeforeModelCallEvent
from strands.plugins import Plugin, hook

logger = logging.getLogger(__name__)

# Rough heuristic: 4 characters per token
CHARS_PER_TOKEN = 4

SUMMARY_PROMPT = """\
You are a conversation compressor. Given the following conversation messages,
produce a structured summary in this exact format:

## Resolved
- (bullet points of what was discussed and completed)

## Pending
- (bullet points of open questions or incomplete tasks)

## Active Task
(one sentence describing what the user is currently working on)

## Key Context
- (any important facts, names, IDs, or decisions that must be preserved)

Be concise but preserve all actionable information. Do NOT include greetings or filler.

CONVERSATION TO COMPRESS:
{conversation}
"""


def _estimate_tokens(content: Any) -> int:
    """Estimate token count for a message or content block."""
    if isinstance(content, str):
        return max(1, len(content) // CHARS_PER_TOKEN)
    if isinstance(content, dict):
        return max(1, len(json.dumps(content, default=str)) // CHARS_PER_TOKEN)
    if isinstance(content, list):
        return sum(_estimate_tokens(item) for item in content)
    return max(1, len(str(content)) // CHARS_PER_TOKEN)


def _estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single conversation message."""
    tokens = 0
    # Role overhead
    tokens += 4
    # Content
    content = message.get("content", [])
    tokens += _estimate_tokens(content)
    return tokens


def _format_messages_for_summary(messages: list[dict]) -> str:
    """Format messages into a readable string for the summarizer."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(block["text"])
                    elif "toolUse" in block:
                        tool_use = block["toolUse"]
                        parts.append(f"[tool_use: {tool_use.get('name', '?')}]")
                    elif "toolResult" in block:
                        tool_result = block["toolResult"]
                        result_content = tool_result.get("content", [])
                        preview = str(result_content)[:200]
                        parts.append(f"[tool_result: {preview}]")
                    else:
                        parts.append(str(block)[:100])
                else:
                    parts.append(str(block)[:100])
            text = "\n".join(parts)
        else:
            text = str(content)[:500]
        lines.append(f"[{role}]: {text[:1000]}")
    return "\n\n".join(lines)


class ContextPlugin(Plugin):
    """Manages context window compression for long conversations.

    Monitors token usage before each model call. When the conversation approaches
    the configured threshold, compresses middle turns into a structured summary
    while protecting the first and last N messages.
    """

    name = "context"

    def __init__(
        self,
        max_context_tokens: int = 100_000,
        threshold_percent: float = 0.75,
        protect_first_n: int = 3,
        protect_last_n: int = 6,
    ):
        """Initialize the context compression plugin.

        Args:
            max_context_tokens: Maximum tokens for the context window.
            threshold_percent: Fraction of max_context_tokens that triggers compression.
            protect_first_n: Number of initial messages to never compress (system prompt + setup).
            protect_last_n: Number of recent messages to never compress (active work).
        """
        self._max_context_tokens = max_context_tokens
        self._threshold_percent = threshold_percent
        self._protect_first_n = protect_first_n
        self._protect_last_n = protect_last_n

        # Stats tracking
        self._compression_count = 0
        self._total_tokens_saved = 0
        self._last_estimated_tokens = 0

        # Agent reference set during init_agent
        self._agent: Agent | None = None

        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Store agent reference for accessing messages during hook execution."""
        self._agent = agent

    @hook
    def check_context_size(self, event: BeforeModelCallEvent) -> None:
        """Before each model call, check if context compression is needed."""
        if self._agent is None:
            return

        messages = self._agent.messages

        if not messages:
            return

        # Estimate current token usage
        total_tokens = sum(_estimate_message_tokens(msg) for msg in messages)
        self._last_estimated_tokens = total_tokens

        # Use projected_input_tokens from the event if available (more accurate)
        if event.projected_input_tokens is not None:
            total_tokens = event.projected_input_tokens
            self._last_estimated_tokens = total_tokens

        threshold = int(self._max_context_tokens * self._threshold_percent)

        if total_tokens < threshold:
            return

        # Need at least protect_first + protect_last + 1 messages to compress anything
        min_messages = self._protect_first_n + self._protect_last_n + 1
        if len(messages) <= min_messages:
            logger.debug(
                "Context over threshold (%d/%d tokens) but not enough messages to compress (%d <= %d)",
                total_tokens,
                threshold,
                len(messages),
                min_messages,
            )
            return

        logger.info(
            "Context compression triggered: %d tokens (threshold: %d). Compressing middle messages.",
            total_tokens,
            threshold,
        )

        self._compress_middle_messages(messages)

    def _compress_middle_messages(self, messages: list[dict]) -> None:
        """Compress middle messages into a structured summary."""
        first_n = self._protect_first_n
        last_n = self._protect_last_n

        # Identify the compressible range
        middle_messages = messages[first_n:-last_n] if last_n > 0 else messages[first_n:]

        if not middle_messages:
            return

        # Estimate tokens being removed
        tokens_before = sum(_estimate_message_tokens(msg) for msg in middle_messages)

        # Generate summary using a sub-agent
        conversation_text = _format_messages_for_summary(middle_messages)
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

        try:
            summarizer = Agent(
                system_prompt="You are a precise conversation summarizer. Output only the structured summary.",
            )
            result = summarizer(prompt)
            summary_text = str(result.message)
        except Exception as e:
            logger.error("Context compression failed during summarization: %s", e)
            # Fallback: create a simple indicator that messages were removed
            summary_text = (
                f"[{len(middle_messages)} messages were compressed. "
                f"Estimated {tokens_before} tokens removed. Summarization failed.]"
            )

        # Build the replacement summary message
        summary_message = {
            "role": "assistant",
            "content": [
                {
                    "text": (
                        f"[CONTEXT SUMMARY - {len(middle_messages)} messages compressed, "
                        f"~{tokens_before} tokens saved]\n\n{summary_text}"
                    )
                }
            ],
        }

        # Replace middle messages in-place
        if last_n > 0:
            messages[first_n:-last_n] = [summary_message]
        else:
            messages[first_n:] = [summary_message]

        # Update stats
        tokens_after = _estimate_message_tokens(summary_message)
        tokens_saved = tokens_before - tokens_after
        self._compression_count += 1
        self._total_tokens_saved += max(0, tokens_saved)

        logger.info(
            "Compressed %d messages into summary. Saved ~%d tokens. Total compressions: %d",
            len(middle_messages),
            tokens_saved,
            self._compression_count,
        )

    @tool
    def get_context_stats(self) -> str:
        """Show current context window usage statistics.

        Returns token estimates, compression history, and configuration.
        """
        if self._agent is None:
            return "Context plugin not attached to an agent."

        messages = self._agent.messages
        message_count = len(messages)
        current_tokens = sum(_estimate_message_tokens(msg) for msg in messages)
        threshold = int(self._max_context_tokens * self._threshold_percent)
        usage_pct = (current_tokens / self._max_context_tokens * 100) if self._max_context_tokens > 0 else 0

        lines = [
            "Context Window Stats:",
            f"  Messages: {message_count}",
            f"  Estimated tokens: {current_tokens:,} / {self._max_context_tokens:,} ({usage_pct:.1f}%)",
            f"  Compression threshold: {threshold:,} tokens ({self._threshold_percent * 100:.0f}%)",
            f"  Protected messages: first {self._protect_first_n}, last {self._protect_last_n}",
            "",
            "Compression History:",
            f"  Times compressed: {self._compression_count}",
            f"  Total tokens saved: ~{self._total_tokens_saved:,}",
            f"  Last estimated input: {self._last_estimated_tokens:,} tokens",
        ]
        return "\n".join(lines)
