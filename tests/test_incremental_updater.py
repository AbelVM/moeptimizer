"""Tests for incremental context updates."""


from moeptimizer.incremental_updater import (
    IncrementalUpdater,
    get_incremental_updater,
)


class TestIncrementalUpdater:
    def test_empty_updater(self) -> None:
        """Empty updater has no context versions."""
        updater = IncrementalUpdater()
        assert updater.get_version([]) == 0

    def test_update_context_new(self) -> None:
        """Update context with new content."""
        updater = IncrementalUpdater()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
        ]
        result = updater.update_context(messages, "New content")
        assert len(result) >= len(messages)

    def test_update_context_appends(self) -> None:
        """Update context appends to known context."""
        updater = IncrementalUpdater()
        messages = [{"role": "user", "content": "Test"}]
        # First call registers the context
        updater.update_context(messages, "")
        # Second call with same context and new content should append
        result = updater.update_context(messages, "Appended")
        assert "Appended" in result[-1].get("content", "")

    def test_should_preserve_cache(self) -> None:
        """Check if cache should be preserved."""
        updater = IncrementalUpdater()
        old = [{"role": "user", "content": "Hello"}]
        new = [{"role": "user", "content": "Hello World"}]
        assert updater.should_preserve_cache(old, new) is True
        # Different content
        other = [{"role": "user", "content": "Different"}]
        assert updater.should_preserve_cache(old, other) is False

    def test_singleton(self) -> None:
        """Get incremental updater returns new instance each time."""
        updater1 = get_incremental_updater()
        updater2 = get_incremental_updater()
        # Function returns new instances
        assert isinstance(updater1, IncrementalUpdater)
        assert isinstance(updater2, IncrementalUpdater)
