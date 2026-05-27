"""Goals plugin: track metrics and progress. PostgreSQL via SQLAlchemy."""

import logging

from sqlalchemy.exc import IntegrityError

from strands import tool
from strands.plugins import Plugin

from pythia_agent.db import Goal, get_session
from pythia_agent.utils import slugify, utc_now

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nYou can track measurable goals with progress over time. "
    "Use goals when the user sets objectives with numeric targets."
)


class GoalsPlugin(Plugin):
    """Provides goal tracking backed by PostgreSQL."""

    name = "goals"

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    def __init__(self):
        super().__init__()

    @tool
    def create_goal(self, name: str, target: float, unit: str = "", period: str = "") -> str:
        """Create a new goal to track progress toward.

        Args:
            name: Name of the goal (e.g., 'Blog posts written')
            target: Target value to reach
            unit: Unit of measurement (e.g., 'posts', 'miles', 'commits')
            period: Time period (e.g., 'weekly', 'monthly', 'Q1 2025')
        """
        goal_id = slugify(name)
        with get_session() as session:
            try:
                session.add(Goal(id=goal_id, name=name, target=target, unit=unit, period=period))
                session.commit()
            except IntegrityError:
                session.rollback()
                return f"Goal '{goal_id}' already exists. Use update_goal to modify it."
        return f"Created goal '{name}' — target: {target} {unit} ({period or 'no period'})"

    @tool
    def update_goal(self, goal_id: str, increment: float = 0, set_value: float = -1) -> str:
        """Update progress on a goal by incrementing or setting its current value.

        Args:
            goal_id: ID of the goal to update
            increment: Amount to add to current value
            set_value: Set current value directly (overrides increment if >= 0)
        """
        with get_session() as session:
            goal = session.get(Goal, goal_id)
            if not goal:
                return f"Goal '{goal_id}' not found."

            goal.current = set_value if set_value >= 0 else goal.current + increment
            goal.updated_at = utc_now()
            session.commit()

            pct = (goal.current / goal.target * 100) if goal.target > 0 else 0
            status = "exceeded" if goal.current >= goal.target else "in progress"
            return f"Goal '{goal.name}': {goal.current}/{goal.target} {goal.unit} ({pct:.0f}%) — {status}"

    @tool
    def check_goals(self) -> str:
        """View all goals and their current progress."""
        with get_session() as session:
            goals = session.query(Goal).order_by(Goal.created_at).all()

        if not goals:
            return "No goals defined."

        lines = [f"Goals ({len(goals)} total):"]
        for g in goals:
            pct = (g.current / g.target * 100) if g.target > 0 else 0
            bar_filled = int(pct / 5)
            bar = "█" * min(bar_filled, 20) + "░" * (20 - min(bar_filled, 20))
            check = "✓" if g.current >= g.target else " "
            period_str = f" [{g.period}]" if g.period else ""
            lines.append(
                f"[{check}] {g.name}{period_str}\n"
                f"    {bar} {g.current}/{g.target} {g.unit} ({pct:.0f}%)"
            )
        return "\n".join(lines)

    @tool
    def delete_goal(self, goal_id: str) -> str:
        """Delete a goal.

        Args:
            goal_id: ID of the goal to delete
        """
        with get_session() as session:
            goal = session.get(Goal, goal_id)
            if not goal:
                return f"Goal '{goal_id}' not found."
            session.delete(goal)
            session.commit()
        return f"Goal '{goal_id}' deleted."
