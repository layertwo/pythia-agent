"""Dreams plugin: exposes memory consolidation as agent-callable tools."""

from __future__ import annotations

import logging

from strands import tool
from strands.plugins import Plugin

from pythia_agent.dreams import DreamEngine, DreamStatus

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nYou can consolidate your long-term memory store on demand with "
    "`dream`. It dedupes, resolves contradictions, and optionally surfaces "
    "new insights from recent session transcripts. The result is applied "
    "immediately; use `rollback_dream` if a dream removed something "
    "important or you don't like the result."
)


class DreamsPlugin(Plugin):
    """Tools for triggering memory consolidation dreams and rolling them back."""

    name = "dreams"

    def __init__(self, engine: DreamEngine, user_id: str):
        self._engine = engine
        self._user_id = user_id
        super().__init__()

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    @tool
    def dream(self, instructions: str = "") -> str:
        """Start a memory consolidation dream: dedupe, resolve contradictions, surface insights.

        Runs in the background on the dreams executor so it doesn't block
        the chat/SSE response. Call `list_dreams` to check progress; once
        the most recent dream reaches `completed`, the new consolidated
        set is live. Use `rollback_dream` if you don't like the result.

        Args:
            instructions: Optional focus area. If provided, recent session
                transcripts are also analyzed so the model can surface new
                insights (otherwise the dream only consolidates existing
                memories).
        """
        self._engine.submit(
            user_id=self._user_id,
            instructions=instructions or None,
            include_sessions=bool(instructions),
            trigger="manual",
        )
        focus = f" Focus: {instructions}." if instructions else ""
        return (
            "Dream queued. It runs in the background and will replace your "
            "memory store with the consolidated set when done. Use "
            "`list_dreams` to check status; `rollback_dream` to undo." + focus
        )

    @tool
    def rollback_dream(self, steps_back: int = 1) -> str:
        """Restore memories to their state before the Nth most recent dream.

        Args:
            steps_back: How many dreams to undo (default 1 = undo the most recent).
        """
        result = self._engine.rollback(user_id=self._user_id, steps_back=steps_back)
        if not result.get("ok"):
            return f"Rollback failed: {result.get('error')}"
        return (
            f"Rolled back to state before dream {result['dream_id']} "
            f"(started {result['dream_started_at']}). "
            f"Restored {result['restored_count']} memories."
        )

    @tool
    def list_dreams(self, limit: int = 10) -> str:
        """List recent memory consolidation dreams for this user with their outcomes.

        Args:
            limit: Maximum number of dreams to list.
        """
        runs = self._engine.list_runs(user_id=self._user_id, limit=limit)
        if not runs:
            return "No dreams have run for this user."

        lines = [f"Recent dreams ({len(runs)}):"]
        for r in runs:
            counts = (
                f"{r['count_before']} → {r['count_after']}"
                if r["count_before"] is not None
                else "—"
            )
            tail = ""
            if r["status"] == DreamStatus.REJECTED_BY_GUARDRAIL.value:
                tail = f" [guardrail: {r['guardrail_reason']}]"
            elif r["status"] == DreamStatus.FAILED.value:
                tail = f" [error: {r['error']}]"
            lines.append(
                f"- {r['started_at']} [{r['status']}] {counts} ({r['trigger']}) "
                f"id={r['id'][:8]}{tail}"
            )
        return "\n".join(lines)
