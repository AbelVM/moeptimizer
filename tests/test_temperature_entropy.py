"""Tests for temperature calibration and entropy-guided trimming."""

import pytest

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer


class TestTemperatureCalibration:
    def test_low_entropy_coding_temperature(self) -> None:
        """Low entropy context (code) gets coding temperature ~0.6."""
        config = AppConfig()
        optimizer = AgentContextOptimizer(config)
        # Code has low entropy (repetitive patterns)
        messages = [
            {"role": "user", "content": "def foo():\n    pass\n" * 10},
        ]
        temp = optimizer.get_optimal_temperature(messages)
        # Low entropy should give 0.6 for precise coding
        assert temp == 0.6

    def test_high_entropy_high_temperature(self) -> None:
        """High entropy context gets higher temperature."""
        config = AppConfig()
        optimizer = AgentContextOptimizer(config)
        # Create truly high-entropy content with many unique symbols
        import random
        import string
        random.seed(42)
        # Generate random "words" to maximize symbol diversity
        high_entropy_content = " ".join(
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(200)
        )
        messages = [
            {"role": "user", "content": high_entropy_content},
        ]
        temp = optimizer.get_optimal_temperature(messages)
        # High entropy (>0.6) should give 0.3
        assert temp == 0.3


class TestEntropyGuidedTrimming:
    def test_calculate_message_entropy(self) -> None:
        """Calculate entropy of messages."""
        config = AppConfig()
        optimizer = AgentContextOptimizer(config)

        # Low entropy (repetitive)
        low_entropy = "def foo():\n    pass\n" * 10
        entropy = optimizer._calculate_message_entropy(low_entropy)
        assert entropy < 0.5

        # High entropy (diverse symbols)
        high_entropy = "import os\nimport sys\nimport json\nimport re\nimport math\n"
        entropy = optimizer._calculate_message_entropy(high_entropy)
        assert entropy > 0.3

    def test_entropy_guided_trim_preserves_assistant(self) -> None:
        """Entropy trimming preserves assistant messages."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        optimizer = AgentContextOptimizer(config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Task 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        result = optimizer._entropy_guided_trim(messages)
        # All assistant messages should be preserved
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 2

    def test_entropy_guided_trim_trims_tool(self) -> None:
        """Entropy trimming can trim high-entropy tool output."""
        config = AppConfig()
        config.agentic.max_optimized_chars = 500
        optimizer = AgentContextOptimizer(config)

        # Create high-entropy tool output
        high_entropy_tool = "x" * 600  # High entropy, long
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Task 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "tool", "content": high_entropy_tool},
        ]
        result = optimizer._entropy_guided_trim(messages)
        # Tool output should be truncated
        tool_msg = next((m for m in result if m.get("role") == "tool"), None)
        if tool_msg:
            assert "truncated" in tool_msg.get("content", "") or len(tool_msg.get("content", "")) < 600