"""Safety plugin: prevents runaway agent behavior with iteration budgets and tool guardrails."""

import hashlib
import logging
import threading
from collections import defaultdict

from strands import tool
from strands.hooks import AfterInvocationEvent, AfterToolCallEvent
from strands.plugins import Plugin, hook

logger = logging.getLogger(__name__)

# Default set of idempotent (read-only) tool names that are safe to repeat.
DEFAULT_IDEMPOTENT_TOOLS: set[str] = frozenset(
    {
        "check_budget",
        "list_jobs",
        "job_history",
        "search_memory",
        "recall",
        "get_goal",
        "list_goals",
        "web_search",
        "read_url",
    }
)


class SafetyPlugin(Plugin):
    """Prevents runaway agent behavior with iteration budgets and repetitive-call detection."""

    name = "safety"

    def __init__(
        self,
        max_iterations: int = 50,
        max_repeats: int = 3,
        idempotent_tools: set[str] | None = None,
    ):
        self._max_iterations = max_iterations
        self._max_repeats = max_repeats
        self._idempotent_tools: set[str] = (
            set(idempotent_tools) if idempotent_tools is not None else set(DEFAULT_IDEMPOTENT_TOOLS)
        )

        self._iteration_count: int = 0
        self._tool_call_counts: dict[str, int] = defaultdict(int)
        self._warnings: list[str] = []
        self._lock = threading.Lock()

        super().__init__()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @hook
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Increment iteration counter and track tool calls for repetition detection."""
        with self._lock:
            self._iteration_count += 1

            # Build a key from tool name + hash of arguments
            tool_name = getattr(event, "tool_name", None) or "unknown"
            tool_args = getattr(event, "tool_arguments", None) or {}
            args_hash = hashlib.sha256(str(sorted(tool_args.items())).encode()).hexdigest()[:16]
            call_key = f"{tool_name}:{args_hash}"

            self._tool_call_counts[call_key] += 1
            count = self._tool_call_counts[call_key]

            # Check for repetitive calls
            if count >= self._max_repeats:
                is_idempotent = tool_name in self._idempotent_tools
                category = "idempotent" if is_idempotent else "mutating"
                warning = (
                    f"Repetitive call detected: '{tool_name}' with same arguments called "
                    f"{count} times ({category}). Consider a different approach."
                )
                if warning not in self._warnings:
                    self._warnings.append(warning)
                    logger.warning(warning)

            # Check budget exhaustion
            if self._iteration_count >= self._max_iterations:
                logger.warning(
                    "Iteration budget exhausted: %d/%d tool calls used. Stopping.",
                    self._iteration_count,
                    self._max_iterations,
                )

    @hook
    def after_invocation(self, event: AfterInvocationEvent) -> None:
        """Reset counters at the end of each invocation."""
        with self._lock:
            self._iteration_count = 0
            self._tool_call_counts.clear()
            self._warnings.clear()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    def check_budget(self) -> str:
        """Check remaining iteration budget and report any detected loops or repetitive tool calls."""
        with self._lock:
            remaining = self._max_iterations - self._iteration_count
            lines = [
                f"Iteration budget: {self._iteration_count}/{self._max_iterations} used, {remaining} remaining.",
            ]

            # Report repetitive calls
            repeated = {k: v for k, v in self._tool_call_counts.items() if v >= self._max_repeats}
            if repeated:
                lines.append("")
                lines.append("Detected repetitive calls:")
                for call_key, count in sorted(repeated.items(), key=lambda x: -x[1]):
                    tool_name = call_key.split(":")[0]
                    is_idempotent = tool_name in self._idempotent_tools
                    category = "idempotent" if is_idempotent else "MUTATING"
                    lines.append(f"  - {tool_name} [{category}]: {count} calls with same args")

            # Report accumulated warnings
            if self._warnings:
                lines.append("")
                lines.append("Warnings this invocation:")
                for w in self._warnings:
                    lines.append(f"  - {w}")

            if not repeated and not self._warnings:
                lines.append("No loops or repetitive calls detected.")

            return "\n".join(lines)
