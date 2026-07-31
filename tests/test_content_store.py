"""Tests for the reversible-compression content store (review §4.1.2 / B1)."""

from moeptimizer.content_store import ContentStore


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
