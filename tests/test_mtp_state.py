"""Tests for MTP state management."""

import pytest

from moeptimizer.mtp_state import (
    MTPStateManager,
    get_mtp_state_manager,
)


class TestMTPStateManager:
    def test_empty_state(self) -> None:
        """Empty state manager has no states."""
        manager = MTPStateManager()
        assert manager.get_stats()["saves"] == 0
        assert manager.get_stats()["loads"] == 0

    def test_save_and_load_state(self) -> None:
        """Save and load state works."""
        manager = MTPStateManager()
        manager.save_state("test_key", {"hidden": [1, 2, 3]})
        result = manager.load_state("test_key")
        assert result == {"hidden": [1, 2, 3]}

    def test_load_missing_state(self) -> None:
        """Load missing state returns None."""
        manager = MTPStateManager()
        result = manager.load_state("nonexistent")
        assert result is None

    def test_get_state_key(self) -> None:
        """Get state key generates consistent keys."""
        manager = MTPStateManager()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        key1 = manager.get_state_key(messages)
        key2 = manager.get_state_key(messages)
        assert key1 == key2
        assert len(key1) == 16  # MD5 truncated to 16 chars

    def test_stats_tracking(self) -> None:
        """Stats are tracked correctly."""
        manager = MTPStateManager()
        manager.save_state("key1", {"data": 1})
        manager.save_state("key2", {"data": 2})
        manager.load_state("key1")  # hit
        manager.load_state("key3")  # miss
        stats = manager.get_stats()
        assert stats["saves"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_clear(self) -> None:
        """Clear removes all states."""
        manager = MTPStateManager()
        manager.save_state("key1", {"data": 1})
        manager.clear()
        assert manager.load_state("key1") is None

    def test_singleton(self) -> None:
        """Get mtp state manager returns singleton."""
        manager1 = get_mtp_state_manager()
        manager2 = get_mtp_state_manager()
        assert manager1 is manager2