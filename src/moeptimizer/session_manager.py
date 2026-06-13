"""SessionManager — Per-session optimizer isolation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from moeptimizer.config import get_config
from moeptimizer.optimizer import AgentContextOptimizer


class SessionManager:
    """
    Manages per-session optimizer instances for concurrent request isolation.

    Each agent session gets its own AgentContextOptimizer with isolated state.
    Sessions expire after AGENTIC_SESSION_TIMEOUT seconds of inactivity.
    """

    def __init__(self, session_timeout: int | None = None) -> None:
        config = get_config().agentic
        self._sessions: dict[str, AgentContextOptimizer] = {}
        self._session_timestamps: dict[str, float] = {}
        self._session_timeout = session_timeout or config.session_timeout

    def get_or_create(self, session_id: str | None = None) -> AgentContextOptimizer:
        """Get an existing optimizer for the session, or create a new one."""
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]

        now = time.time()
        self._cleanup_expired(now)

        if session_id not in self._sessions:
            self._sessions[session_id] = AgentContextOptimizer()
            self._session_timestamps[session_id] = now

        self._session_timestamps[session_id] = now
        return self._sessions[session_id]

    def get_state(self, session_id: str) -> str | None:
        """Get serialized state for a session."""
        optimizer = self._sessions.get(session_id)
        if optimizer:
            return optimizer.get_session_state()
        return None

    def load_state(self, session_id: str, state_json: str) -> bool:
        """Load state into a session's optimizer."""
        optimizer = self._sessions.get(session_id)
        if optimizer:
            try:
                optimizer.load_session_state(state_json)
                return True
            except Exception:
                return False
        return False

    def reset_session(self, session_id: str) -> bool:
        """Reset a session's optimizer."""
        if session_id in self._sessions:
            self._sessions[session_id] = AgentContextOptimizer()
            self._session_timestamps[session_id] = time.time()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            del self._session_timestamps[session_id]
            return True
        return False

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        """List all active sessions with metadata."""
        result: dict[str, dict[str, Any]] = {}
        now = time.time()
        for sid, optimizer in self._sessions.items():
            goal = optimizer.store.get_goal()
            result[sid] = {
                "step_count": len(optimizer.store.steps),
                "goal": goal.original_prompt[:100] if goal else None,
                "last_active": self._session_timestamps.get(sid, 0),
                "age_seconds": now - self._session_timestamps.get(sid, now),
            }
        return result

    def _cleanup_expired(self, now: float) -> None:
        """Remove sessions that have exceeded the timeout."""
        expired = [
            sid for sid, ts in self._session_timestamps.items()
            if now - ts > self._session_timeout
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._session_timestamps.pop(sid, None)
