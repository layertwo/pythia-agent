"""Memory consolidation pipeline ("dreams").

Reads a user's current memories (and optionally recent session transcripts),
asks an LLM to emit consolidation operations against a strict Pydantic
schema, then atomically replaces the user's memories with the consolidated
set. Full snapshot of the pre-dream memories is stored on `Dream` for
rollback. Old Dream rows beyond `retain_runs` are pruned.

Modeled on Anthropic's Managed Agents "Dreams" feature
(https://platform.claude.com/docs/en/managed-agents/dreams), adapted for a
self-hosted single-tenant agent: no review/promote workflow, auto-applied
with a diff-size guardrail and a tool-driven rollback escape hatch.

Auto-fire is gated by BOTH a cron schedule AND minimum activity since the
last dream — inspired by claudefa.st/blog/guide/mechanics/auto-dream's
"dual-gate" idea (don't dream if nothing's happened; don't dream too often).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from croniter import croniter
from pydantic import BaseModel, Discriminator, Field
from sqlalchemy import desc

from pythia_agent.config import DreamsConfig, MemoryConfig
from pythia_agent.db import Dream, get_session
from pythia_agent.memory import Mem0SessionManager
from pythia_agent.models.session import ConversationSession, SessionMessage
from pythia_agent.providers.factory import create_model
from pythia_agent.utils import utc_now

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Operation schema — Strands' structured_output enforces this at the
# model layer via Ollama's grammar/format param.
# ----------------------------------------------------------------------


class DreamStatus(str, Enum):
    """Lifecycle states for a Dream row.

    Transitions:
        RUNNING ──► APPLYING ──► COMPLETED
                              └► (crash) ──► INTERRUPTED (via startup sweep)
        RUNNING ──► REJECTED_BY_GUARDRAIL  (no destructive op ran)
        RUNNING ──► FAILED                 (exception in pipeline)

    Inherits from `str` so SQLAlchemy stores the plain string value and
    existing JSON/DB rows are interchangeable with enum members.
    """

    RUNNING = "running"
    APPLYING = "applying"
    COMPLETED = "completed"
    REJECTED_BY_GUARDRAIL = "rejected_by_guardrail"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class KeepOp(BaseModel):
    action: Literal["keep"] = "keep"
    memory_id: str


class MergeOp(BaseModel):
    action: Literal["merge"] = "merge"
    memory_ids: list[str] = Field(min_length=2)
    new_text: str
    reason: str


class ReplaceOp(BaseModel):
    action: Literal["replace"] = "replace"
    memory_id: str
    new_text: str
    reason: str


class DropOp(BaseModel):
    action: Literal["drop"] = "drop"
    memory_id: str
    reason: str


class InsightOp(BaseModel):
    action: Literal["insight"] = "insight"
    new_text: str
    source_sessions: list[str] = Field(min_length=1)


Operation = Annotated[
    KeepOp | MergeOp | ReplaceOp | DropOp | InsightOp,
    Discriminator("action"),
]


class DreamResult(BaseModel):
    operations: list[Operation]


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are a memory curator for an AI assistant called Pythia. Your job is
to consolidate Pythia's long-term memory store about a single
user. The store accumulates across many sessions; over time it
collects duplicates, contradictions, and stale entries.

Today's date is {today}. Use this to judge what counts as "stale" or
"transient state".

You will be given:
- The current memory store as a JSON list. Each memory has an `id` and `text`.
- (Optional) Recent session transcripts where the assistant and user spoke,
  as a JSON list. Each session has a `session_id`, `started_at`, and `messages[]`.
  Use these to ground INSIGHT operations and to verify contradictions.
- (Optional) `instructions` from the user about what to focus on.

If no sessions are provided, only KEEP / MERGE / REPLACE / DROP are valid;
do not emit INSIGHT without a session to cite.

For each memory in the store, emit exactly one operation:

- KEEP — the memory is accurate, distinct from others, and still useful.
- MERGE — this memory overlaps with one or more others; consolidate
  them into a single canonical statement.
- REPLACE — this memory is contradicted by newer information in the
  sessions or by another memory; emit a corrected version.
- DROP — this memory is stale, no longer relevant, or rendered
  redundant by a MERGE you're emitting elsewhere.

Additionally, you may emit any number of:

- INSIGHT — a new memory worth storing, derived from patterns observed
  across the session transcripts but not currently captured.

## Rules

1. Every MERGE must list every source memory id being combined.
2. Every REPLACE and DROP must give a one-line reason.
3. Every INSIGHT must cite at least one session id supporting it.
4. **Do not fabricate.** If a fact is not clearly supported by the
   input memories or sessions, do not emit it.
5. **Bias toward KEEP.** When in doubt, leave the memory alone.
   Forgetting something real is worse than keeping mild redundancy.
6. **Preserve specificity.** "User runs Python 3.14 on macOS" is more
   useful than "User likes programming." When merging, do not lose detail.
7. **Resolve contradictions toward the most recent evidence.**
8. Memories about user preferences, identity, or stable facts are
   higher value than transient state ("currently debugging X").
   Lean toward dropping old transient state.
9. **Rewrite relative dates to absolute.** When you KEEP, MERGE, REPLACE,
   or emit an INSIGHT about a memory containing words like "today",
   "yesterday", "last week", "two months ago", "this month", etc., rewrite
   the relative reference to an absolute date computed from today's date
   above and the memory's original context. Relative dates stored
   verbatim become wrong the moment time passes; absolute dates stay true.
"""


_USER_PROMPT = """\
Current memories:
{memories_json}

Recent sessions:
{sessions_json}

User instructions: {instructions}
"""


def _build_prompt(
    memories: list[dict],
    sessions: list[dict],
    instructions: str | None,
    today: str,
) -> tuple[str, str]:
    system = _SYSTEM_PROMPT.format(today=today)
    user = _USER_PROMPT.format(
        memories_json=json.dumps(memories, indent=2, default=str),
        sessions_json=(
            json.dumps(sessions, indent=2, default=str) if sessions else "(none — perform consolidation only)"
        ),
        instructions=instructions or "(none — perform routine consolidation)",
    )
    return system, user


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


class DreamEngine:
    """Runs memory consolidation dreams and persists outcomes.

    Per-instance executor (max_workers=1) so concurrent triggers (cron +
    on-demand tool) serialize naturally; per-instance state means tests
    and reload scenarios that build multiple DreamEngines stay isolated.
    """

    def __init__(self, dreams_config: DreamsConfig, memory_config: MemoryConfig, model_settings):
        self.dreams_config = dreams_config
        self.memory_config = memory_config
        self.model_settings = model_settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dream")
        self._cron_thread: threading.Thread | None = None
        self._cron_stop = threading.Event()
        self._started_cron = False
        # Cache Mem0SessionManager per user_id so concurrent triggers reuse
        # the same warm mem0.Memory client (spaCy + pgvector pool + embedder
        # init are expensive enough to amortize across dreams).
        self._session_managers: dict[str, Mem0SessionManager] = {}
        self._sm_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def submit(self, user_id: str, **kwargs) -> None:
        """Queue a dream to run on the background executor."""
        self._executor.submit(self._do_dream_safe, user_id, **kwargs)

    # Statuses that are safe to roll back to: the dream is no longer mutating
    # the store. `applying` is excluded — a concurrent rollback against an
    # in-flight wipe would interleave delete/add loops and corrupt the store.
    # `interrupted` is included because the startup sweep reclassifies any
    # stale `running`/`applying` rows there.
    _ROLLBACK_STATUSES = (
        DreamStatus.COMPLETED,
        DreamStatus.REJECTED_BY_GUARDRAIL,
        DreamStatus.FAILED,
        DreamStatus.INTERRUPTED,
    )

    def rollback(self, user_id: str, steps_back: int = 1) -> dict:
        """Restore the memory store to its state before the Nth most recent dream.

        Accepts any dream that has a `memories_before` snapshot AND is in a
        terminal state. Excludes `applying` so a rollback can't race with an
        in-flight wipe.
        """
        if steps_back < 1:
            return {"ok": False, "error": "steps_back must be >= 1"}

        with get_session() as db:
            runs = (
                db.query(Dream)
                .filter(
                    Dream.user_id == user_id,
                    Dream.memories_before.isnot(None),
                    Dream.status.in_(self._ROLLBACK_STATUSES),
                )
                .order_by(desc(Dream.started_at))
                .limit(steps_back)
                .all()
            )
            if len(runs) < steps_back:
                return {
                    "ok": False,
                    "error": f"only {len(runs)} restorable dreams available for user '{user_id}'",
                }
            target = runs[-1]
            snapshot = target.memories_before or []
            target_id = target.id
            target_started = target.started_at

        # Restore via mem0: wipe current, re-add snapshot.
        sm = self._session_manager(user_id)
        self._replace_user_memories(sm, snapshot)
        logger.info("Rolled back user '%s' to state before dream %s", user_id, target_id)
        return {
            "ok": True,
            "dream_id": target_id,
            "dream_started_at": target_started.isoformat() if target_started else None,
            "restored_count": len(snapshot),
        }

    def list_runs(self, user_id: str, limit: int = 10) -> list[dict]:
        with get_session() as db:
            runs = (
                db.query(Dream)
                .filter(Dream.user_id == user_id)
                .order_by(desc(Dream.started_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "status": r.status,
                    "trigger": r.trigger,
                    "count_before": r.count_before,
                    "count_after": r.count_after,
                    "guardrail_reason": r.guardrail_reason,
                    "error": r.error,
                }
                for r in runs
            ]

    # ------------------------------------------------------------------
    # Cron
    # ------------------------------------------------------------------

    def start_cron(self) -> None:
        if not self.dreams_config.enabled:
            logger.info("Dreams cron disabled (dreams.enabled=false)")
            return
        if not croniter.is_valid(self.dreams_config.cron):
            logger.error("Invalid dreams cron: %s — cron disabled", self.dreams_config.cron)
            return
        # Heal first so the dual gate doesn't treat stale `applying`/`running`
        # rows from a prior crashed process as recent dreams (which would
        # silently block this user's cron forever).
        self._heal_stale_runs()
        self._cron_stop.clear()
        self._cron_thread = threading.Thread(
            target=self._cron_loop, name="dreams-cron", daemon=True
        )
        self._cron_thread.start()
        self._started_cron = True
        logger.info("Dreams cron started: cron='%s'", self.dreams_config.cron)

    def _heal_stale_runs(self) -> None:
        """Reclassify any `running`/`applying` rows older than 1h as `interrupted`.

        A row in those states at engine start means the previous process was
        killed mid-dream. Leaving them blocks cron via the dual-gate filter
        and confuses `list_dreams`. The 1h floor avoids racing with a live
        dream in another worker on the off chance there is one.
        """
        cutoff = utc_now() - timedelta(hours=1)
        with get_session() as db:
            stale = (
                db.query(Dream)
                .filter(
                    Dream.status.in_((DreamStatus.RUNNING, DreamStatus.APPLYING)),
                    Dream.started_at < cutoff,
                )
                .all()
            )
            for d in stale:
                d.status = DreamStatus.INTERRUPTED
                d.completed_at = utc_now()
                d.error = "process restart while dream was in-flight"
            if stale:
                db.commit()
                logger.warning(
                    "Reclassified %d stale dream(s) (running/applying > 1h) to 'interrupted'",
                    len(stale),
                )

    def shutdown(self) -> None:
        """Stop the cron loop and drain pending background dreams. Called from app lifespan shutdown."""
        self._cron_stop.set()
        if self._cron_thread:
            self._cron_thread.join(timeout=5)
        self._executor.shutdown(wait=True)

    def _cron_loop(self) -> None:
        # Seed last_fire from the most recent prior cron-triggered dream so
        # restarts don't swallow missed fires. With no prior runs, seed at
        # the previous scheduled fire (epoch-anchored), which forces the
        # first tick to fire if the current cycle's slot has passed.
        last_fire = self._initial_last_fire()
        while not self._cron_stop.is_set():
            try:
                now = utc_now()
                next_fire = croniter(self.dreams_config.cron, last_fire).get_next(datetime)
                if next_fire <= now:
                    last_fire = next_fire
                    self._fire_for_all_users(now)
            except Exception:
                logger.exception("Dreams cron tick failed")
            self._cron_stop.wait(60.0)

    def _fire_for_all_users(self, now: datetime) -> None:
        """Submit a dream for every user whose dual-gate passes.

        Auto-discovers users from `ConversationSession.user_id`. Each user
        gets their own dual-gate check, so quiet users don't waste LLM
        calls and active users dream on their own cadence. Cron dreams
        always include sessions so INSIGHTs are valid.
        """
        users = self._users_with_activity()
        if not users:
            logger.info("Cron tick: no users with conversation activity, skipping")
            return
        for user_id in users:
            gate = self._dual_gate_check(user_id, now)
            if gate is None:
                self.submit(user_id, include_sessions=True, trigger="cron")
            else:
                logger.info("Skipping cron dream for user '%s': %s", user_id, gate)

    @staticmethod
    def _users_with_activity() -> list[str]:
        """Distinct user_ids that have at least one ConversationSession."""
        with get_session() as db:
            return [r[0] for r in db.query(ConversationSession.user_id).distinct().all()]

    def _dual_gate_check(self, user_id: str, now: datetime) -> str | None:
        """Return a reason string if the dream should be skipped, else None.

        Inspired by claudefa.st/auto-dream: a dream is only worth running if
        enough TIME has passed AND enough ACTIVITY has occurred since the
        last dream. Otherwise we either dream too often (wasted LLM calls
        on a quiet user) or trigger on a single long session split over
        multiple days (cron fires but nothing's actually changed).
        """
        with get_session() as db:
            # Include `running` so a still-queued dream from a prior cron tick
            # blocks duplicate enqueueing under cloud rate-limit backlog. The
            # startup sweep reclassifies stale `running`/`applying` rows so
            # they don't block forever after a process crash.
            last = (
                db.query(Dream)
                .filter(
                    Dream.user_id == user_id,
                    Dream.status.in_(
                        (
                            DreamStatus.RUNNING,
                            DreamStatus.APPLYING,
                            DreamStatus.COMPLETED,
                            DreamStatus.REJECTED_BY_GUARDRAIL,
                        )
                    ),
                )
                .order_by(desc(Dream.started_at))
                .first()
            )
            if last is None:
                return None  # first ever — let it fire

            since = now - last.started_at
            min_hours = self.dreams_config.min_hours_between_dreams
            if since < timedelta(hours=min_hours):
                return f"only {since.total_seconds() / 3600:.1f}h since last dream (need {min_hours}h)"

            session_count = (
                db.query(SessionMessage)
                .join(ConversationSession, SessionMessage.session_id == ConversationSession.id)
                .filter(
                    ConversationSession.user_id == user_id,
                    SessionMessage.created_at > last.started_at,
                )
                .with_entities(ConversationSession.id)
                .distinct()
                .count()
            )
            min_sessions = self.dreams_config.min_sessions_between_dreams
            if session_count < min_sessions:
                return f"only {session_count} sessions since last dream (need {min_sessions})"
        return None

    def _initial_last_fire(self) -> datetime:
        """Decide what `last_fire` to start the cron loop with after boot.

        Anchors on the most recent cron-triggered dream across ALL users —
        the cron schedule itself is a global tick; per-user gating happens
        inside `_fire_for_all_users` via the dual gate.
        """
        with get_session() as db:
            prior = (
                db.query(Dream)
                .filter(Dream.trigger == "cron")
                .order_by(desc(Dream.started_at))
                .first()
            )
        if prior and prior.started_at:
            return prior.started_at
        # No prior cron run — anchor just before the most recent scheduled
        # fire so croniter.get_next from this seed returns that fire (which
        # is <= now if we deployed past the fire window) and the loop runs
        # a catch-up on first tick. Without the epsilon, get_next would
        # return tomorrow's fire and today's would silently be lost.
        prev_fire = croniter(self.dreams_config.cron, utc_now()).get_prev(datetime)
        return prev_fire - timedelta(microseconds=1)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _do_dream_safe(self, user_id: str, **kwargs) -> None:
        try:
            self._do_dream(user_id, **kwargs)
        except Exception:
            logger.exception("Dream failed for user_id=%s", user_id)

    def _do_dream(
        self,
        user_id: str,
        instructions: str | None = None,
        include_sessions: bool = False,
        trigger: str = "manual",
    ) -> dict:
        run_id = str(uuid.uuid4())
        started_at = utc_now()
        logger.info("Dream %s starting (user=%s, trigger=%s)", run_id, user_id, trigger)

        row_inserted = False
        try:
            # Insert running row. Inside the try so a DB blip during the
            # initial commit is logged + propagated rather than producing an
            # orphan exception with no audit trail.
            with get_session() as db:
                db.add(
                    Dream(
                        id=run_id,
                        user_id=user_id,
                        started_at=started_at,
                        status=DreamStatus.RUNNING,
                        trigger=trigger,
                        instructions=instructions,
                    )
                )
                db.commit()
            row_inserted = True

            sm = self._session_manager(user_id)
            memories = sm.get_all() or []
            mem_inputs = [{"id": m.get("id"), "text": m.get("memory", "")} for m in memories]

            sessions = self._collect_sessions(user_id) if include_sessions else []

            today = datetime.now(timezone.utc).date().isoformat()
            system_prompt, user_prompt = _build_prompt(mem_inputs, sessions, instructions, today)

            result = self._call_llm(system_prompt, user_prompt)
            consolidated = self._apply_operations(mem_inputs, result)

            count_before = len(mem_inputs)
            count_after = len(consolidated)
            guardrail = self._check_guardrail(result, count_before, count_after)

            if guardrail:
                self._finalize(
                    run_id,
                    status=DreamStatus.REJECTED_BY_GUARDRAIL,
                    memories_before=mem_inputs,
                    operations=result.model_dump()["operations"],
                    count_before=count_before,
                    count_after=count_after,
                    guardrail_reason=guardrail,
                )
                logger.warning("Dream %s rejected by guardrail: %s", run_id, guardrail)
                return {
                    "ok": False,
                    "dream_id": run_id,
                    "status": DreamStatus.REJECTED_BY_GUARDRAIL.value,
                    "reason": guardrail,
                    "count_before": count_before,
                    "count_after": count_after,
                }

            # Persist the snapshot + planned ops BEFORE the destructive op so
            # a crash mid-wipe leaves a recoverable Dream behind. Rollback
            # accepts any run with memories_before set, not just "completed".
            self._finalize(
                run_id,
                status=DreamStatus.APPLYING,
                memories_before=mem_inputs,
                operations=result.model_dump()["operations"],
                count_before=count_before,
                count_after=count_after,
            )

            # Wipe current user memories, write consolidated set. Not truly
            # atomic across mem0 + pgvector — if the process dies mid-loop
            # the user's store is partially destroyed, but the snapshot
            # above lets `rollback_dream` restore from before.
            self._replace_user_memories(sm, consolidated)

            self._finalize(run_id, status=DreamStatus.COMPLETED)
            self._prune_old_runs(user_id)
            logger.info(
                "Dream %s completed: %d → %d memories", run_id, count_before, count_after
            )
            return {
                "ok": True,
                "dream_id": run_id,
                "status": DreamStatus.COMPLETED.value,
                "count_before": count_before,
                "count_after": count_after,
            }

        except Exception as e:
            logger.exception("Dream %s failed", run_id)
            if row_inserted:
                self._finalize(run_id, status=DreamStatus.FAILED, error=str(e))
            return {
                "ok": False,
                "dream_id": run_id,
                "status": DreamStatus.FAILED.value,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> DreamResult:
        """Run the consolidation prompt with the DreamResult schema enforced.

        Uses the non-deprecated `structured_output_model=` parameter on the
        agent invocation rather than the deprecated `Agent.structured_output`
        method. Strands' agent dispatch handles the format/grammar setup.
        """
        from strands import Agent

        model = create_model(self.model_settings)
        agent = Agent(model=model, system_prompt=system_prompt)
        result = agent(user_prompt, structured_output_model=DreamResult)
        if result.structured_output is None:
            raise RuntimeError("LLM returned no structured output for DreamResult")
        return result.structured_output

    # ------------------------------------------------------------------
    # Operation application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_operations(memories: list[dict], result: DreamResult) -> list[str]:
        """Translate the LLM's operations into a final list of memory texts.

        Validates that every input memory is addressed by exactly one
        operation. Conflicting operations on the same id (e.g. KEEP+DROP,
        MERGE listing an id also DROPped, duplicate ids across MERGEs)
        would otherwise yield order-dependent output; we resolve them
        deterministically: the FIRST op wins and subsequent conflicting
        ops on the same id are skipped with a warning. Unknown ids
        referenced by MERGE are dropped from that op.

        Returns just the text strings; ids are not preserved across an
        dream (mem0 assigns fresh ids on re-insert).
        """
        by_id = {m["id"]: m["text"] for m in memories if m.get("id")}
        consolidated: list[str] = []
        consumed: set[str] = set()

        def claim(mem_id: str, op_name: str) -> bool:
            """Mark `mem_id` consumed. Return False if already claimed."""
            if mem_id in consumed:
                logger.warning(
                    "Dream op %s references memory %s already addressed earlier; skipping",
                    op_name,
                    mem_id,
                )
                return False
            if mem_id not in by_id:
                logger.warning(
                    "Dream op %s references unknown memory id %s; skipping",
                    op_name,
                    mem_id,
                )
                return False
            consumed.add(mem_id)
            return True

        for op in result.operations:
            if isinstance(op, KeepOp):
                if claim(op.memory_id, "KEEP"):
                    consolidated.append(by_id[op.memory_id])
            elif isinstance(op, MergeOp):
                claimed_any = False
                for mid in op.memory_ids:
                    if claim(mid, "MERGE"):
                        claimed_any = True
                if claimed_any:
                    consolidated.append(op.new_text)
            elif isinstance(op, ReplaceOp):
                if claim(op.memory_id, "REPLACE"):
                    consolidated.append(op.new_text)
            elif isinstance(op, DropOp):
                claim(op.memory_id, "DROP")
            elif isinstance(op, InsightOp):
                consolidated.append(op.new_text)

        # Defensive keep: any input memory the model didn't address survives.
        # Pairs with the "bias toward KEEP" instruction in the prompt.
        for mem_id, text in by_id.items():
            if mem_id not in consumed:
                consolidated.append(text)
                logger.warning("Dream response omitted memory %s; keeping defensively", mem_id)

        # Dedupe identical strings while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for text in consolidated:
            key = text.strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(text)
        return deduped

    # ------------------------------------------------------------------
    # Guardrail
    # ------------------------------------------------------------------

    def _check_guardrail(
        self, result: DreamResult, count_before: int, count_after: int
    ) -> str | None:
        """Return a reason string if the dream should be rejected, else None.

        Three checks over the input memory set:

        - `drop_ratio` — share of inputs the model intentionally removed
          via DropOp. MERGEs are intentional dedupe, not destructive drops,
          so they don't count here.
        - `rewrite_ratio` — share of inputs the model replaced in place
          via ReplaceOp. MERGEs are also excluded — collapsing 5 paraphrases
          of "I use Python" into one statement is the feature working, not
          mass rewriting.
        - `retention_ratio` — final size vs. input size. Catches the case
          where MERGE-exclusion above creates a blind spot: a model emitting
          only aggressive MergeOps (50 specific memories → 2 generic
          summaries) trips zero drop/rewrite ratios but destroys signal.
        """
        if count_before == 0:
            return None

        drops = sum(1 for op in result.operations if isinstance(op, DropOp))
        replaces = sum(1 for op in result.operations if isinstance(op, ReplaceOp))

        drop_ratio = drops / count_before
        rewrite_ratio = replaces / count_before
        retention_ratio = count_after / count_before

        if drop_ratio > self.dreams_config.max_drop_ratio:
            return f"drop_ratio={drop_ratio:.2f} > {self.dreams_config.max_drop_ratio}"
        if rewrite_ratio > self.dreams_config.max_rewrite_ratio:
            return f"rewrite_ratio={rewrite_ratio:.2f} > {self.dreams_config.max_rewrite_ratio}"
        if retention_ratio < self.dreams_config.min_retention_ratio:
            return (
                f"retention_ratio={retention_ratio:.2f} < "
                f"{self.dreams_config.min_retention_ratio}"
            )
        return None

    # ------------------------------------------------------------------
    # Memory mutation helpers
    # ------------------------------------------------------------------

    def _session_manager(self, user_id: str) -> Mem0SessionManager:
        with self._sm_lock:
            sm = self._session_managers.get(user_id)
            if sm is None:
                sm = self._session_managers[user_id] = Mem0SessionManager(self.memory_config, user_id=user_id)
            return sm

    @staticmethod
    def _replace_user_memories(sm: Mem0SessionManager, texts: list[str]) -> None:
        """Wipe all memories for this user, then re-add the consolidated set.

        We let mem0 run its normal extraction pipeline (`infer` left at the
        default) so the `pythia_memories_entities` table gets repopulated —
        bypassing it via `infer=False` writes the vector but skips
        `_link_entities_for_memory`, breaking PR #19 entity-aware retrieval.
        The cost is one extra LLM call per consolidated text; acceptable
        for nightly batch work.
        """
        existing = sm.get_all() or []
        for mem in existing:
            mid = mem.get("id")
            if mid:
                try:
                    sm.delete(mid)
                except Exception:
                    logger.warning("Failed to delete memory %s during dream replace", mid)

        for text in texts:
            try:
                sm.add([{"role": "user", "content": text}])
            except Exception:
                logger.exception("Failed to re-insert memory during dream replace")

    # ------------------------------------------------------------------
    # Session collection
    # ------------------------------------------------------------------

    def _collect_sessions(self, user_id: str) -> list[dict]:
        with get_session() as db:
            sessions = (
                db.query(ConversationSession)
                .filter(ConversationSession.user_id == user_id)
                .order_by(desc(ConversationSession.updated_at))
                .limit(self.dreams_config.max_sessions)
                .all()
            )
            out: list[dict] = []
            for s in sessions:
                msgs = (
                    db.query(SessionMessage)
                    .filter(SessionMessage.session_id == s.id)
                    .order_by(SessionMessage.created_at)
                    .limit(self.dreams_config.max_messages_per_session)
                    .all()
                )
                out.append(
                    {
                        "session_id": s.id,
                        "started_at": s.created_at.isoformat() if s.created_at else None,
                        "messages": [{"role": m.role, "text": m.content} for m in msgs],
                    }
                )
            return out

    # ------------------------------------------------------------------
    # DB plumbing
    # ------------------------------------------------------------------

    def _finalize(
        self,
        run_id: str,
        *,
        status: DreamStatus,
        memories_before: list[dict] | None = None,
        operations: list[dict[str, Any]] | None = None,
        count_before: int | None = None,
        count_after: int | None = None,
        guardrail_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        with get_session() as db:
            run = db.get(Dream, run_id)
            if not run:
                return
            run.status = status
            run.completed_at = utc_now()
            if memories_before is not None:
                run.memories_before = memories_before
            if operations is not None:
                run.operations = operations
            if count_before is not None:
                run.count_before = count_before
            if count_after is not None:
                run.count_after = count_after
            if guardrail_reason is not None:
                run.guardrail_reason = guardrail_reason
            if error is not None:
                run.error = error
            db.commit()

    def _prune_old_runs(self, user_id: str) -> None:
        keep = max(self.dreams_config.retain_runs, 1)
        with get_session() as db:
            runs = (
                db.query(Dream)
                .filter(Dream.user_id == user_id)
                .order_by(desc(Dream.started_at))
                .all()
            )
            for stale in runs[keep:]:
                db.delete(stale)
            db.commit()
