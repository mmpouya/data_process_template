"""In-memory agent sessions for human-in-the-loop question/answer flow.

Each running agent owns one :class:`AgentSession`. The `ask_user` tool blocks
on the session's answer queue until the user replies through the separate
`/chat/answer` endpoint, which forwards the answer via :class:`SessionManager`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.config import get_settings
from .schema import ResultMessage

logger = logging.getLogger(__name__)


class AgentSession:
    """
    Per-run state shared between the streaming agent and the answer API.

    """

    # actually it works like an `structs` (C) or a pydantic Basemodel with the time methods.

    def __init__(self, publish: Callable[[ResultMessage], Awaitable[None]]) -> None:
        self.session_id: str = uuid.uuid4().hex
        self.created_at: float = time.monotonic()
        self.last_activity: float = time.monotonic()
        self.publish = publish
        self.answer_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending_question_id: str | None = None
        self.work_details: list[dict[str, Any]] = []

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def is_expired(self, ttl: int | None = None) -> bool:
        ttl = ttl if ttl is not None else get_settings().SESSION_TTL
        return time.monotonic() - self.last_activity > ttl


class SessionManager:
    """Registry of live sessions with lazy TTL-based garbage collection.

    `SessionManager` is the only bridge between the two separate HTTP requests that make HITL work.
    The stream run and the answer submission are different requests/tasks with no shared call stack,
    so a shared lookup keyed by `session_id` is the only way to route an answer back into the still-blocked run.

    - **Routing answers to a blocked run:**
    `ask_user` blocks on `session.answer_queue.get()` (per AGENTS.md §2 step 6).
    The `/chat/answer` endpoint can only reach that queue through `SessionManager.submit_answer` (session.py:74-97) — without the registry,
    the answer has nowhere to go and HITL is impossible.

    - **Lifecycle ownership:**
    `create` (session.py:55-63) registers the session at stream start; `release` (session.py:99-103) runs in the run's `finally`; `_gc` (session.py:105-116) lazily evicts sessions whose `release` never happened (client disconnect, crash) using `SESSION_TTL` — otherwise sessions/queues leak forever.

    - **Concurrency safety:**
    `_sessions` is class-level shared state (session.py:51-52) touched by concurrent tasks (create/get/submit/release/GC); the `asyncio.Lock` prevents interleaved mutations.

    - **Enforcing the HITL invariant:**
    `submit_answer` rejects missing/already-answered/stale `question_id` submissions (session.py:79-91), guaranteeing at most one pending question per session — a mismatched or duplicate answer is dropped rather than corrupting the run.
    """

    _sessions: dict[str, AgentSession] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def create(
        cls, publish: Callable[[ResultMessage], Awaitable[None]]
    ) -> AgentSession:
        await cls._gc()
        session = AgentSession(publish)
        async with cls._lock:  # EXPLANATION: Without the lock, operations such as inserting, removing, and looking up sessions could overlap
            cls._sessions[session.session_id] = session
        logger.debug("session created: %s", session.session_id)
        return session

    @classmethod
    async def get(cls, session_id: str) -> AgentSession | None:
        await cls._gc()
        async with cls._lock:
            session = cls._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    @classmethod
    async def submit_answer(
        cls, session_id: str, question_id: str | None, answer: Any
    ) -> bool:
        async with cls._lock:
            session = cls._sessions.get(session_id)
            if session is None:
                return False
            if session.pending_question_id is None:
                return False
            if question_id and question_id != session.pending_question_id:
                logger.debug(
                    "stale answer dropped for %s (pending=%s, got=%s)",
                    session_id,
                    session.pending_question_id,
                    question_id,
                )
                return False

            session.pending_question_id = None
            await session.answer_queue.put(
                {"question_id": question_id, "answer": answer}
            )
            return True

    @classmethod
    async def release(cls, session_id: str) -> None:
        async with cls._lock:
            cls._sessions.pop(session_id, None)
        logger.debug("session released: %s", session_id)

    @classmethod
    async def _gc(cls) -> None:
        # _gc stands for Garbage collector. collecter? i don't know.
        async with cls._lock:
            expired = [
                session_id
                for session_id, session in cls._sessions.items()
                if session.is_expired()
            ]
            for session_id in expired:
                cls._sessions.pop(session_id, None)
                logger.debug("session expired: %s", session_id)
