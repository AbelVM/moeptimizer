"""Tests for sliding window context with MTP state preservation."""


from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


class TestSlidingWindowContext:
    def test_sliding_window_trim_small_context(self) -> None:
        """Small context is not trimmed."""
        config = AppConfig()
        optimizer = AgentContextOptimizer(config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = optimizer._sliding_window_trim(messages, window_size=4096)
        assert len(result) == len(messages)

    def test_sliding_window_trim_large_context(self) -> None:
        """Large context drops whole old turns while preserving active request."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 10000
        # Disable cache-stable mode so this test exercises pure sliding-window
        # eviction (the frozen early turns would otherwise be immutable).
        config.v050.cache_stable_mode = False
        optimizer = AgentContextOptimizer(config)
        # Create enough old dynamic turns that only complete turns can be evicted.
        large_content = "x" * 800
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First user"},
        ]
        for index in range(40):
            messages.append({"role": "assistant", "content": f"Response {index}"})
            messages.append({"role": "user", "content": f"Task {index}"})
        messages.append({"role": "user", "content": large_content})

        result = optimizer._sliding_window_trim(messages, window_size=1200)

        # Static prefix and the newest user turn are immutable; only complete
        # old dynamic turns are dropped from the front.
        expected = messages[:2] + messages[42:]
        assert result == expected
        assert len(result) < len(messages)
        assert result[-1].get("content") == large_content
        assert all(
            len(m.get("content", "")) == len(e.get("content", ""))
            for m, e in zip(result, expected, strict=True)
        )

    def test_sliding_window_preserves_static_layer(self) -> None:
        """Sliding window preserves static layer (system + first user)."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 10000
        optimizer = AgentContextOptimizer(config)
        large_content = "x" * 2000
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "First user"},
            {"role": "user", "content": large_content},
        ]
        result = optimizer._sliding_window_trim(messages, window_size=100)
        # Static layer should be preserved
        assert result[0].get("role") == "system"
        assert result[1].get("role") == "user"

    def test_sliding_window_preserves_suffix_order(self) -> None:
        """Sliding window keeps the newest dynamic suffix in chronological order."""
        config = AppConfig()
        optimizer = AgentContextOptimizer(config)
        # Disable cache-stable mode: this test checks pure suffix-ordering of the
        # dynamic layer, not the frozen early-turn prefix.
        config.v050.cache_stable_mode = False
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "three"},
        ]
        result = optimizer._sliding_window_trim(messages, window_size=20)
        roles = [msg.get("role") for msg in result]
        assert roles == ["system", "user", "assistant"]
        assert result[-1].get("content") == "three"
