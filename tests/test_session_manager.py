"""Tests for SessionManager."""

import time

from moeptimizer.session_manager import SessionManager


class TestSessionManager:
    def test_create_session(self) -> None:
        manager = SessionManager(session_timeout=3600)
        optimizer = manager.get_or_create("test-session")
        assert optimizer is not None
        sessions = manager.list_sessions()
        assert "test-session" in sessions

    def test_get_existing_session(self) -> None:
        manager = SessionManager(session_timeout=3600)
        opt1 = manager.get_or_create("test-session")
        opt2 = manager.get_or_create("test-session")
        assert opt1 is opt2

    def test_auto_generate_session_id(self) -> None:
        manager = SessionManager(session_timeout=3600)
        optimizer = manager.get_or_create()
        assert optimizer is not None

    def test_reset_session(self) -> None:
        manager = SessionManager(session_timeout=3600)
        opt1 = manager.get_or_create("test-session")
        manager.reset_session("test-session")
        opt2 = manager.get_or_create("test-session")
        assert opt1 is not opt2

    def test_delete_session(self) -> None:
        manager = SessionManager(session_timeout=3600)
        manager.get_or_create("test-session")
        assert manager.delete_session("test-session")
        assert not manager.delete_session("test-session")

    def test_session_timeout(self) -> None:
        manager = SessionManager(session_timeout=1)
        manager.get_or_create("test-session")
        time.sleep(1.1)
        manager.get_or_create("other-session")  # triggers cleanup
        sessions = manager.list_sessions()
        assert "test-session" not in sessions

    def test_get_state(self) -> None:
        manager = SessionManager(session_timeout=3600)
        manager.get_or_create("test-session")
        state = manager.get_state("test-session")
        assert state is not None
        assert "store" in state

    def test_max_sessions_lru_eviction(self) -> None:
        manager = SessionManager(session_timeout=3600)
        manager._max_sessions = 3
        # Create 3 sessions, then touch s0 so it is most-recently-active.
        for sid in ("s0", "s1", "s2"):
            manager.get_or_create(sid)
        manager.get_or_create("s0")  # bump s0's timestamp
        # Adding a 4th session evicts the LRU (s1), not the recently-used s0.
        manager.get_or_create("s3")
        sessions = manager.list_sessions()
        assert set(sessions) == {"s0", "s2", "s3"}
        assert "s1" not in sessions

    def test_max_sessions_cap_disabled(self) -> None:
        manager = SessionManager(session_timeout=3600)
        manager._max_sessions = 0  # disabled
        for i in range(20):
            manager.get_or_create(f"s{i}")
        assert len(manager.list_sessions()) == 20
