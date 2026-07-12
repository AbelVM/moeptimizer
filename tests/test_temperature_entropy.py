"""Tests for temperature calibration and entropy-guided trimming."""

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


class TestMessageEntropy:
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
