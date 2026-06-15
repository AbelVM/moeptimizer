"""Tests for hit_prediction_model module."""

import pytest

from moeptimizer.hit_prediction_model import HitPredictionModel, get_hit_prediction_model


class TestHitPredictionModel:
    def setup_method(self) -> None:
        self.model = HitPredictionModel(max_history=50, retrain_threshold=10)

    def test_extract_features_basic(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        features = self.model.extract_features(messages)
        assert "total_tokens" in features
        assert "static_ratio" in features
        assert features["message_count"] == 3.0

    def test_extract_features_with_code(self) -> None:
        messages = [
            {"role": "user", "content": "Fix this:\n```python\ndef foo():\n    pass\n```"},
        ]
        features = self.model.extract_features(messages)
        assert features["code_block_count"] == 1.0

    def test_extract_features_with_tool_calls(self) -> None:
        messages = [
            {"role": "assistant", "content": "Let me check", "tool_calls": [{"id": "1"}]},
        ]
        features = self.model.extract_features(messages)
        assert features["has_tool_calls"] == 1.0

    def test_record_outcome(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        self.model.record_outcome(messages, hit=True)
        assert len(self.model._history) == 1

    def test_predict_hit_rate_fallback(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        # Without training, should use heuristic
        rate = self.model.predict_hit_rate(messages)
        assert 0.0 <= rate <= 1.0

    def test_should_early_exit_high_confidence(self) -> None:
        # Create messages with high static ratio (likely cache hit)
        messages = [
            {"role": "system", "content": "x" * 1000},
            {"role": "user", "content": "y"},
        ]
        # Should not early exit with default threshold
        result = self.model.should_early_exit(messages, threshold=0.99)
        assert isinstance(result, bool)

    def test_should_aggressive_trim(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = self.model.should_aggressive_trim(messages, threshold=0.99)
        assert isinstance(result, bool)

    def test_train_insufficient_data(self) -> None:
        result = self.model.train()
        assert result is False  # Not enough data

    def test_get_stats(self) -> None:
        stats = self.model.get_stats()
        assert "predictions" in stats
        assert "early_exits" in stats

    def test_reset(self) -> None:
        self.model.record_outcome([{"role": "user", "content": "hi"}], hit=True)
        self.model.reset()
        assert len(self.model._history) == 0
        assert self.model._trained is False

    def test_global_instance(self) -> None:
        model = get_hit_prediction_model()
        assert isinstance(model, HitPredictionModel)
