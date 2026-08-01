"""Tests for the reversible-compression content store (review §4.1.2 / B1)."""

from moeptimizer.content_store import ContentStore, extract_placeholder_handle


class TestContentStore:
    def test_put_get_round_trip(self) -> None:
        store = ContentStore()
        original = "def f():\n    return 42\n" * 100
        handle = store.put(original)
        assert store.get(handle) == original

    def test_handle_is_deterministic(self) -> None:
        store = ContentStore()
        content = "the same content"
        assert store.handle(content) == store.handle(content)
        assert store.put(content) == store.put(content)  # idempotent

    def test_get_unknown_returns_none(self) -> None:
        store = ContentStore()
        assert store.get("does-not-exist") is None

    def test_extract_placeholder_handle(self) -> None:
        assert extract_placeholder_handle(
            "[original retained: handle=abc123def4567890, 42 chars; use expand_content]"
        ) == "abc123def4567890"
        assert extract_placeholder_handle("ordinary output") is None

    def test_stores_are_isolated(self) -> None:
        first = ContentStore()
        second = ContentStore()
        handle = first.put("private output")
        assert second.get(handle) is None

    def test_lru_eviction(self) -> None:
        store = ContentStore(max_entries=2)
        h1 = store.put("one")
        store.put("two")
        store.put("three")  # evicts the LRU ("one")
        assert store.get(h1) is None
        assert len(store) == 2

    def test_get_refreshes_recency(self) -> None:
        store = ContentStore(max_entries=2)
        h1 = store.put("one")
        store.put("two")
        assert store.get(h1) == "one"  # refreshes h1 -> "two" is now LRU
        store.put("three")  # evicts "two", not h1
        assert store.get(h1) == "one"
        assert len(store) == 2
