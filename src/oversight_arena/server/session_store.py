"""Session storage for the Oversight Arena HTTP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import uuid4

from oversight_arena.environment import OversightArenaEnv


DEFAULT_SESSION_TTL_SECONDS = 900
UtcNow = Callable[[], datetime]


class SessionStoreError(Exception):
    """Base exception for session-store failures."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a requested session handle does not exist."""


class SessionExpiredError(SessionNotFoundError):
    """Raised when a requested session handle has expired."""


@dataclass(slots=True)
class SessionRecord:
    """One active or recently completed server-side environment session."""

    session_id: str
    env: OversightArenaEnv
    created_at: datetime
    expires_at: datetime

    def refresh(self, *, now: datetime, ttl_seconds: int) -> None:
        """Extend this session's expiry window from the supplied timestamp."""

        self.expires_at = now + timedelta(seconds=ttl_seconds)


class SessionStore:
    """Keep per-session environment state out of the FastAPI route handlers."""

    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        now: UtcNow | None = None,
    ) -> None:
        """Create an in-memory session store with sliding-expiry semantics."""

        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._now = now or utc_now
        self._sessions: dict[str, SessionRecord] = {}

    @property
    def ttl_seconds(self) -> int:
        """Return the configured session time-to-live in seconds."""

        return self._ttl_seconds

    def create(self, env: OversightArenaEnv) -> SessionRecord:
        """Register a reset environment under a fresh session handle."""

        self.cleanup_expired_sessions()
        current_time = self._now()
        record = SessionRecord(
            session_id=uuid4().hex,
            env=env,
            created_at=current_time,
            expires_at=current_time + timedelta(seconds=self._ttl_seconds),
        )
        self._sessions[record.session_id] = record
        return record

    def get(self, session_id: str, *, refresh: bool = True) -> SessionRecord:
        """Return one session record or raise an explicit session error."""

        self.cleanup_expired_sessions()
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFoundError(f"unknown session_id: {session_id}")

        current_time = self._now()
        if record.expires_at <= current_time:
            del self._sessions[session_id]
            raise SessionExpiredError(f"expired session_id: {session_id}")

        if refresh:
            record.refresh(now=current_time, ttl_seconds=self._ttl_seconds)
        return record

    def delete(self, session_id: str) -> None:
        """Remove one session handle if it exists."""

        self._sessions.pop(session_id, None)

    def cleanup_expired_sessions(self) -> int:
        """Delete expired sessions and return the number removed."""

        current_time = self._now()
        expired_ids = [
            session_id
            for session_id, record in self._sessions.items()
            if record.expires_at <= current_time
        ]
        for session_id in expired_ids:
            del self._sessions[session_id]
        return len(expired_ids)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


__all__ = [
    "DEFAULT_SESSION_TTL_SECONDS",
    "SessionExpiredError",
    "SessionNotFoundError",
    "SessionRecord",
    "SessionStore",
    "SessionStoreError",
    "utc_now",
]
