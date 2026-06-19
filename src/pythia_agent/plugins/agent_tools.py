"""Agent tools plugin: think (deep reasoning), use_llm (sub-agent), stop, journal."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from strands import Agent, tool
from strands.plugins import Plugin

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nUse think for complex reasoning before acting. Use journal to log daily work and outcomes. "
    "Use use_llm to delegate isolated subtasks to a fresh agent context."
)


class AgentToolsPlugin(Plugin):
    """Provides deep thinking, sub-agent delegation, stop, and journaling tools."""

    name = "agent-tools"

    def __init__(self, journal_dir: str | None = None):
        self.journal_dir = Path(journal_dir) if journal_dir else Path.cwd() / "journal"
        super().__init__()

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    @tool
    def think(self, thought: str, depth: str = "standard") -> str:
        """Process a thought deeply through structured analytical reasoning.

        Use this when you need to reason through a complex problem step by step
        before responding. The output is your internal analysis, not shown to the user.

        Args:
            thought: The problem or question to reason about
            depth: Reasoning depth - 'quick' for surface analysis, 'standard' for thorough, 'deep' for exhaustive
        """
        # This tool's value is in prompting the model to reason explicitly.
        # The model writes its analysis as the return value, which enters
        # its own context for subsequent reasoning.
        prompts = {
            "quick": "Briefly analyze this in 2-3 sentences:",
            "standard": "Analyze this systematically. Consider assumptions, implications, and alternatives:",
            "deep": (
                "Perform exhaustive analysis. Consider: 1) Core assumptions and whether they hold, "
                "2) Multiple perspectives, 3) Edge cases, 4) Second-order effects, "
                "5) What could go wrong, 6) Final synthesis:"
            ),
        }
        prefix = prompts.get(depth, prompts["standard"])
        return f"{prefix}\n\n{thought}\n\n[Analysis complete - use these insights in your response]"

    @tool
    def use_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Delegate a task to a fresh sub-agent instance.

        Creates an isolated agent with its own context to handle a specific subtask.
        Useful for: summarization, translation, format conversion, or any task
        that benefits from a clean context.

        Args:
            prompt: The task or question for the sub-agent
            system_prompt: Optional system prompt to configure the sub-agent's behavior
        """
        try:
            sub_agent = Agent(
                system_prompt=system_prompt or "You are a helpful assistant. Be concise and direct.",
            )
            result = sub_agent(prompt)
            return str(result.message)
        except Exception as e:
            return f"Sub-agent error: {e}"

    @tool
    def stop(self, reason: str = "Task complete") -> str:
        """Signal that the current task is complete and no further action is needed.

        Use this when you have fully answered the user's question or completed
        their request and no additional tool calls or responses are necessary.

        Args:
            reason: Brief explanation of why processing is stopping
        """
        return f"[STOP] {reason}"

    @tool
    def journal_write(self, content: str, date: str = "") -> str:
        """Write or append to a daily journal entry.

        Journal entries are stored as markdown files organized by date.

        Args:
            content: Text content to add to the journal entry
            date: Date for the entry in YYYY-MM-DD format (defaults to today)
        """
        target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.journal_dir / f"{target_date}.md"

        try:
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            entry = f"\n\n## {timestamp}\n\n{content}"

            if not file_path.exists():
                entry = f"# Journal - {target_date}{entry}"

            with file_path.open("a") as f:
                f.write(entry)

            return f"Journal entry added to {file_path}"
        except Exception as e:
            return f"Error writing journal: {e}"

    @tool
    def journal_read(self, date: str = "") -> str:
        """Read a journal entry for a specific date.

        Args:
            date: Date to read in YYYY-MM-DD format (defaults to today)
        """
        target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_path = self.journal_dir / f"{target_date}.md"

        if not file_path.exists():
            return f"No journal entry for {target_date}."

        try:
            return file_path.read_text()
        except Exception as e:
            return f"Error reading journal: {e}"

    @tool
    def journal_list(self, limit: int = 10) -> str:
        """List available journal entries.

        Args:
            limit: Maximum number of entries to list (most recent first)
        """
        if not self.journal_dir.exists():
            return "No journal entries yet."

        entries = sorted(self.journal_dir.glob("*.md"), reverse=True)[:limit]
        if not entries:
            return "No journal entries yet."

        lines = [f"Journal entries ({len(entries)} shown):"]
        for entry in entries:
            size = entry.stat().st_size
            lines.append(f"- {entry.stem} ({size} bytes)")
        return "\n".join(lines)
