"""SessionManager — Per-session optimizer isolation."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from moeptimizer.config import AppConfig, get_config
from moeptimizer.optimizer import AgentContextOptimizer


class SessionManager:
    """
    Manages per-session optimizer instances for concurrent request isolation.

    Each agent session gets its own AgentContextOptimizer with isolated state.
    Sessions expire after AGENTIC_SESSION_TIMEOUT seconds of inactivity.
    """

    def __init__(
        self,
        session_timeout: int | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        agentic = self._config.agentic
        self._sessions: dict[str, AgentContextOptimizer] = {}
        self._session_timestamps: dict[str, float] = {}
        self._session_timeout = session_timeout or agentic.session_timeout
        self._lock = threading.RLock()

    def _make_optimizer(self) -> AgentContextOptimizer:
        return AgentContextOptimizer(self._config)

    def get_or_create(self, session_id: str | None = None) -> AgentContextOptimizer:
        """Get an existing optimizer for the session, or create a new one."""
        if session_id is None:
            session_id = uuid.uuid4().hex[:12]

        with self._lock:
            now = time.time()
            self._cleanup_expired(now)

            if session_id not in self._sessions:
                self._sessions[session_id] = self._make_optimizer()
                self._session_timestamps[session_id] = now

            self._session_timestamps[session_id] = time.time()
            return self._sessions[session_id]

    def get_state(self, session_id: str) -> str | None:
        """Get serialized state for a session."""
        with self._lock:
            optimizer = self._sessions.get(session_id)
            if optimizer:
                return optimizer.get_session_state()
            return None

    def load_state(self, session_id: str, state_json: str) -> bool:
        """Load state into a session's optimizer."""
        with self._lock:
            optimizer = self._sessions.get(session_id)
            if optimizer:
                try:
                    optimizer.load_session_state(state_json)
                    self._session_timestamps[session_id] = time.time()
                    return True
                except Exception:
                    return False
            return False

    def reset_session(self, session_id: str) -> bool:
        """Reset a session's optimizer."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id] = self._make_optimizer()
                self._session_timestamps[session_id] = time.time()
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                del self._session_timestamps[session_id]
                return True
            return False

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        """List all active sessions with metadata."""
        with self._lock:
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

    def get_mtp_state_key(self, session_id: str) -> str | None:
        """Get the MTP state key for a session."""
        with self._lock:
            optimizer = self._sessions.get(session_id)
            if optimizer:
                return getattr(optimizer, "_last_mtp_state_key", None)
            return None

    def restore_mtp_state(self, session_id: str) -> Any | None:
        """Restore MTP state for a session if available."""
        with self._lock:
            optimizer = self._sessions.get(session_id)
            if optimizer:
                state_key = getattr(optimizer, "_last_mtp_state_key", None)
                if state_key:
                    from moeptimizer.mtp_state import get_mtp_state_manager
                    mtp_manager = get_mtp_state_manager()
                    return mtp_manager.load_state(state_key)
            return None
