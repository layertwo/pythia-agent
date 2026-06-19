"""Tasks plugin: in-session task decomposition and tracking.

In-memory task list the agent uses to decompose complex work into trackable steps.
Tasks persist only for the duration of a session/invocation.
"""

import json
import logging
from dataclasses import dataclass
from typing import List

from strands import tool
from strands.plugins import Plugin

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

STATUS_INDICATORS = {
    "pending": "⏳",
    "in_progress": "\U0001f504",
    "completed": "✅",
    "cancelled": "❌",
}


@dataclass
class Task:
    """A single tracked task."""

    id: str
    content: str
    status: str = "pending"

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}")


GUIDANCE = (
    "\n\nFor multi-step tasks, decompose work into tracked steps using update_tasks. "
    "Check items off as you complete them. This helps maintain focus across long operations."
)


class TasksPlugin(Plugin):
    """Provides in-session task decomposition and tracking.

    Maintains an ordered in-memory task list. List order represents priority.
    Every mutation returns the full current list so the agent always sees state.
    """

    name = "tasks"

    def __init__(self):
        super().__init__()
        self._tasks: List[Task] = []

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    def _format_tasks(self) -> str:
        """Format the current task list as human-readable text."""
        if not self._tasks:
            return "No tasks. Use update_tasks to create a task list."

        lines = [f"Tasks ({len(self._tasks)} total):"]
        for i, task in enumerate(self._tasks, 1):
            indicator = STATUS_INDICATORS.get(task.status, "?")
            lines.append(f"  {i}. {indicator} [{task.id}] {task.content} ({task.status})")

        # Summary counts
        counts = {}
        for task in self._tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
        summary_parts = []
        for status in ["completed", "in_progress", "pending", "cancelled"]:
            if status in counts:
                summary_parts.append(f"{counts[status]} {status}")
        lines.append(f"\nSummary: {', '.join(summary_parts)}")

        return "\n".join(lines)

    @tool
    def update_tasks(self, tasks: str, merge: bool = True) -> str:
        """Update the task list. Use this to decompose work into trackable steps.

        Args:
            tasks: JSON string of task list. Each task is an object with fields:
                   id (str), content (str), status (str: pending|in_progress|completed|cancelled).
                   Example: [{"id": "research", "content": "Research the API", "status": "pending"}]
            merge: If True (default), update existing tasks by id and append new ones.
                   If False, replace the entire list with the provided tasks.
        """
        try:
            task_data = json.loads(tasks)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON — {e}"

        if not isinstance(task_data, list):
            return "Error: tasks must be a JSON array of task objects."

        # Validate all tasks before applying
        parsed_tasks = []
        for item in task_data:
            if not isinstance(item, dict):
                return f"Error: Each task must be an object, got: {type(item).__name__}"
            missing = [f for f in ("id", "content", "status") if f not in item]
            if missing:
                return f"Error: Task missing required fields: {', '.join(missing)}. Got: {item}"
            status = item["status"]
            if status not in VALID_STATUSES:
                return (
                    f"Error: Invalid status '{status}' for task '{item['id']}'. "
                    f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
                )
            parsed_tasks.append(Task(id=item["id"], content=item["content"], status=status))

        if merge:
            # Update existing tasks by id, append new ones
            existing_ids = {t.id: i for i, t in enumerate(self._tasks)}
            for task in parsed_tasks:
                if task.id in existing_ids:
                    self._tasks[existing_ids[task.id]] = task
                else:
                    self._tasks.append(task)
        else:
            # Replace entire list
            self._tasks = parsed_tasks

        return self._format_tasks()

    @tool
    def get_tasks(self) -> str:
        """View the current task list with status indicators and summary."""
        return self._format_tasks()

    @tool
    def clear_tasks(self) -> str:
        """Clear all tasks from the list."""
        count = len(self._tasks)
        self._tasks = []
        return f"Cleared {count} task(s). Task list is now empty."
