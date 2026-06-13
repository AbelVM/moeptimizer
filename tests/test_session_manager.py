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
