"""Shared SQLAlchemy engine, session factory, and models."""

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    cron = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class JobRun(Base):
    __tablename__ = "job_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True))
    status = Column(String, nullable=False, default="running")
    output = Column(Text)


class Goal(Base):
    __tablename__ = "goals"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    target = Column(Float, nullable=False)
    current = Column(Float, nullable=False, default=0)
    unit = Column(String, nullable=False, default="")
    period = Column(String, nullable=False, default="")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class Dream(Base):
    """Tracks one memory consolidation dream for a user.

    `memories_before` snapshots the full pre-dream memory set so the
    `rollback_dream` tool can restore it; we keep the last N runs per user
    (config-driven).
    """

    __tablename__ = "dreams"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))
    status = Column(String, nullable=False, default="running")
    trigger = Column(String, nullable=False, default="manual")  # "manual" | "cron"
    instructions = Column(Text)
    memories_before = Column(JSONB)  # list[{"id": str, "text": str}]
    operations = Column(JSONB)       # raw DreamResult.operations list
    count_before = Column(Integer)
    count_after = Column(Integer)
    guardrail_reason = Column(Text)
    error = Column(Text)


class Persona(Base):
    __tablename__ = "personas"

    slug = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    system_prompt = Column(Text, nullable=False)
    description = Column(String, nullable=False, default="")
    skills = Column(JSONB, nullable=False, default=list)
    active = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ------------------------------------------------------------------
# Engine / Session
# ------------------------------------------------------------------


def get_engine():
    global _engine
    if _engine is None:
        dsn = os.environ.get("DATABASE_URL", "postgresql://pythia:pythia@localhost:5432/pythia")
        _engine = create_engine(dsn, pool_size=5, max_overflow=10, pool_pre_ping=True)
        logger.info("SQLAlchemy engine created: %s", dsn.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


def get_session() -> Session:
    """Get a new session. Caller is responsible for closing it (use as context manager)."""
    return get_session_factory()()


def init_schema() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(get_engine())
    logger.info("Database schema initialized")


def close_engine() -> None:
    """Dispose the engine (call on shutdown)."""
    global _engine, _SessionFactory
    if _engine:
        _engine.dispose()
        _engine = None
        _SessionFactory = None
