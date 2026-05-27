"""Sessions plugin: persist conversation history to PostgreSQL."""

import logging
import uuid
from datetime import datetime, timezone

from strands import tool
from strands.hooks.events import AfterInvocationEvent, MessageAddedEvent
from strands.plugins import Plugin, hook

from pythia_agent.db import get_session
from pythia_agent.models.session import ConversationSession, SessionMessage

logger = logging.getLogger(__name__)


class SessionsPlugin(Plugin):
    """Provides full conversation persistence and session management backed by PostgreSQL."""

    name = "sessions"

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_session_id: str | None = None
        super().__init__()

    def _ensure_session(self) -> str:
        """Ensure a current session exists, creating one if needed."""
        if self.current_session_id is not None:
            return self.current_session_id

        session_id = str(uuid.uuid4())
        with get_session() as db:
            db.add(
                ConversationSession(
                    id=session_id,
                    user_id=self.user_id,
                    title=None,
                    status="active",
                )
            )
            db.commit()

        self.current_session_id = session_id
        logger.info("Created new session: %s for user: %s", session_id, self.user_id)
        return session_id

    @hook
    def on_message_added(self, event: MessageAddedEvent) -> None:
        """Persist every message added to the conversation."""
        message = event.message
        role = message.get("role", "unknown")

        # Extract text content from the message
        content_parts = message.get("content", [])
        text_pieces: list[str] = []
        for part in content_parts:
            if isinstance(part, str):
                text_pieces.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_pieces.append(part["text"])

        content_text = "\n".join(text_pieces)
        if not content_text.strip():
            return

        session_id = self._ensure_session()

        with get_session() as db:
            db.add(
                SessionMessage(
                    session_id=session_id,
                    role=role,
                    content=content_text,
                )
            )
            # Update message count
            session = db.get(ConversationSession, session_id)
            if session:
                session.message_count = (session.message_count or 0) + 1
                session.updated_at = datetime.now(timezone.utc)
            db.commit()

    @hook
    def on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Update session metadata after each invocation completes."""
        if self.current_session_id is None:
            return

        with get_session() as db:
            session = db.get(ConversationSession, self.current_session_id)
            if session:
                session.updated_at = datetime.now(timezone.utc)
                # Auto-generate title from first user message if not set
                if not session.title:
                    first_msg = (
                        db.query(SessionMessage)
                        .filter(
                            SessionMessage.session_id == self.current_session_id,
                            SessionMessage.role == "user",
                        )
                        .order_by(SessionMessage.created_at)
                        .first()
                    )
                    if first_msg and first_msg.content:
                        session.title = first_msg.content[:100]
                db.commit()

    @tool
    def search_sessions(self, query: str) -> str:
        """Search across all session messages using text matching.

        Args:
            query: Search query to find in conversation history
        """
        with get_session() as db:
            # Use ILIKE for broad text search compatibility
            pattern = f"%{query}%"
            results = (
                db.query(SessionMessage)
                .filter(
                    SessionMessage.session_id.in_(
                        db.query(ConversationSession.id).filter(
                            ConversationSession.user_id == self.user_id
                        )
                    ),
                    SessionMessage.content.ilike(pattern),
                )
                .order_by(SessionMessage.created_at.desc())
                .limit(20)
                .all()
            )

        if not results:
            return f"No messages found matching '{query}'."

        lines = [f"Found {len(results)} message(s) matching '{query}':"]
        for msg in results:
            preview = msg.content[:120].replace("\n", " ")
            lines.append(
                f"  [{msg.role}] (session: {msg.session_id[:8]}..., "
                f"{msg.created_at.strftime('%Y-%m-%d %H:%M')}): {preview}"
            )
        return "\n".join(lines)

    @tool
    def list_sessions(self, limit: int = 10) -> str:
        """List recent conversation sessions with title and message count.

        Args:
            limit: Maximum number of sessions to return (default 10)
        """
        with get_session() as db:
            sessions = (
                db.query(ConversationSession)
                .filter(ConversationSession.user_id == self.user_id)
                .order_by(ConversationSession.updated_at.desc())
                .limit(limit)
                .all()
            )

        if not sessions:
            return "No sessions found."

        lines = [f"Recent sessions ({len(sessions)}):"]
        for s in sessions:
            title = s.title or "(untitled)"
            status_icon = "●" if s.status == "active" else "○"
            lines.append(
                f"  {status_icon} {s.id[:8]}... | {title} | "
                f"{s.message_count} msgs | {s.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )
        return "\n".join(lines)

    @tool
    def resume_session(self, session_id: str) -> str:
        """Resume a previous session by returning its recent messages for context injection.

        Args:
            session_id: The session ID to resume (full UUID or prefix)
        """
        with get_session() as db:
            # Support partial ID matching
            session = db.get(ConversationSession, session_id)
            if not session:
                # Try prefix match
                session = (
                    db.query(ConversationSession)
                    .filter(
                        ConversationSession.id.like(f"{session_id}%"),
                        ConversationSession.user_id == self.user_id,
                    )
                    .first()
                )

            if not session:
                return f"Session '{session_id}' not found."

            # Get the last N messages for context
            messages = (
                db.query(SessionMessage)
                .filter(SessionMessage.session_id == session.id)
                .order_by(SessionMessage.created_at.desc())
                .limit(20)
                .all()
            )

        if not messages:
            return f"Session '{session.id[:8]}...' has no messages."

        # Set the current session to the resumed one
        self.current_session_id = session.id

        # Return messages in chronological order
        messages.reverse()
        lines = [
            f"Resumed session: {session.title or '(untitled)'} "
            f"(id: {session.id[:8]}..., {session.message_count} total messages)",
            "--- Last messages ---",
        ]
        for msg in messages:
            lines.append(f"[{msg.role}]: {msg.content}")

        return "\n".join(lines)
