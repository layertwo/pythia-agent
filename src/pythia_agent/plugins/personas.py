"""Personas plugin: manage agent personalities. PostgreSQL via SQLAlchemy."""

import logging

from sqlalchemy.exc import IntegrityError

from strands import tool
from strands.plugins import Plugin

from pythia_agent.db import Persona, get_session
from pythia_agent.utils import slugify, utc_now

logger = logging.getLogger(__name__)


GUIDANCE = (
    "\n\nYou can switch between personas — specialized personalities with different system prompts. "
    "Use personas when the user needs a different mode of operation."
)


class PersonasPlugin(Plugin):
    """Manages agent personas backed by PostgreSQL."""

    name = "personas"

    def __init__(self):
        super().__init__()

    def init_agent(self, agent) -> None:
        agent.system_prompt += GUIDANCE

    def get_active_persona(self) -> Persona | None:
        with get_session() as session:
            return session.query(Persona).filter(Persona.active.is_(True)).first()  # noqa: E712

    @tool
    def create_persona(self, name: str, system_prompt: str, description: str = "", skills: str = "") -> str:
        """Create a new agent persona with a distinct personality.

        Args:
            name: Display name (e.g., 'Research Assistant', 'Code Reviewer')
            system_prompt: System prompt defining this persona's behavior
            description: Brief description of what this persona does
            skills: Comma-separated skill tags this persona specializes in
        """
        slug = slugify(name)
        skill_list = [s.strip() for s in skills.split(",") if s.strip()] if skills else []

        with get_session() as session:
            try:
                session.add(
                    Persona(
                        slug=slug,
                        name=name,
                        system_prompt=system_prompt,
                        description=description,
                        skills=skill_list,
                    )
                )
                session.commit()
            except IntegrityError:
                session.rollback()
                return f"Persona '{slug}' already exists."
        return f"Created persona '{name}' (slug: {slug})"

    @tool
    def list_personas(self) -> str:
        """List all available personas."""
        with get_session() as session:
            personas = session.query(Persona).order_by(Persona.name).all()

        if not personas:
            return "No personas defined."

        lines = [f"Personas ({len(personas)} total):"]
        for p in personas:
            status = "active" if p.active else "inactive"
            skills = p.skills if isinstance(p.skills, list) else []
            skills_str = f" — skills: {', '.join(skills)}" if skills else ""
            lines.append(f"- [{status}] {p.name} (slug: {p.slug}){skills_str}")
            if p.description:
                lines.append(f"  {p.description}")
        return "\n".join(lines)

    @tool
    def switch_persona(self, slug: str) -> str:
        """Activate a persona (deactivates all others).

        Args:
            slug: Slug of the persona to activate
        """
        with get_session() as session:
            target = session.get(Persona, slug)
            if not target:
                return f"Persona '{slug}' not found."

            now = utc_now()
            session.query(Persona).filter(Persona.active.is_(True)).update(  # noqa: E712
                {"active": False, "updated_at": now}
            )
            target.active = True
            target.updated_at = now
            session.commit()
            return f"Switched to persona '{target.name}'. System prompt applies on next invocation."

    @tool
    def get_persona(self, slug: str) -> str:
        """Get full details of a persona including its system prompt.

        Args:
            slug: Slug of the persona to view
        """
        with get_session() as session:
            p = session.get(Persona, slug)
            if not p:
                return f"Persona '{slug}' not found."

            skills = p.skills if isinstance(p.skills, list) else []
            return (
                f"Name: {p.name}\n"
                f"Slug: {p.slug}\n"
                f"Status: {'active' if p.active else 'inactive'}\n"
                f"Description: {p.description or '(none)'}\n"
                f"Skills: {', '.join(skills) if skills else '(none)'}\n"
                f"Created: {p.created_at}\n"
                f"\nSystem Prompt:\n{p.system_prompt}"
            )

    @tool
    def delete_persona(self, slug: str) -> str:
        """Delete a persona.

        Args:
            slug: Slug of the persona to delete
        """
        with get_session() as session:
            p = session.get(Persona, slug)
            if not p:
                return f"Persona '{slug}' not found."
            session.delete(p)
            session.commit()
        return f"Persona '{slug}' deleted."
