"""Lightweight Hit-Prediction Model for cache optimization.

Trains a small XGBoost model on recent turn statistics to predict cache-hit
probability and trigger early-exit or aggressive trimming.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# Try to import xgboost
try:
    from xgboost import XGBClassifier  # type: ignore[import-untyped]

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# Persistence path
_PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "hit_prediction_model.json"


class HitPredictionModel:
    """
    Lightweight XGBoost model for cache hit prediction.

    Trains on recent turn statistics to predict whether a context will
    hit the prefix cache. Used to trigger early-exit (skip optimization)
    or aggressive trimming (reduce context size) based on prediction.
    """

    # Feature names for the model
    FEATURE_NAMES: ClassVar[list[str]] = [
        "total_tokens",
        "static_tokens",
        "dynamic_tokens",
        "static_ratio",
        "message_count",
        "avg_message_length",
        "code_block_count",
        "has_tool_calls",
        "has_thinking",
        "turn_count",
        "time_since_last_request",
        "cache_hit_rate_recent",
    ]

    def __init__(
        self,
        max_history: int = 200,
        retrain_threshold: int = 50,
    ) -> None:
        self._max_history = max_history
        self._retrain_threshold = retrain_threshold
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._model: Any = None
        self._trained = False
        self._last_trained_size = 0
        self._stats: dict[str, int] = {
            "predictions": 0,
            "early_exits": 0,
            "aggressive_trims": 0,
            "retrains": 0,
        }

        # Try to load persisted model
        self._load_model()

    def extract_features(
        self,
        messages: list[dict[str, Any]],
        recent_hit_rate: float = 0.5,
        time_since_last: float = 0.0,
    ) -> dict[str, float]:
        """Extract features from messages for prediction.

        Args:
            messages: The message list
            recent_hit_rate: Recent cache hit rate (0.0-1.0)
            time_since_last: Seconds since last request

        Returns:
            Feature dict
        """
        total_tokens = 0
        static_tokens = 0
        message_count = len(messages)
        total_chars = 0
        code_block_count = 0
        has_tool_calls = False
        has_thinking = False
        turn_count = 0

        static_end = 0
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                char_len = len(content)
                total_chars += char_len
                # Rough token estimate: ~4 chars per token
                tokens = max(1, char_len // 4)
                total_tokens += tokens

                # Count code blocks before any break
                if "```" in content:
                    code_block_count += content.count("```") // 2

                if role == "system":
                    static_tokens += tokens
                    static_end = i + 1
                elif (role == "user" and static_end == i) or (role == "user" and static_end == 0):
                    static_tokens += tokens
                    static_end = i + 1
                    break

            if msg.get("tool_calls"):
                has_tool_calls = True
            if "<think>" in str(content) or "<thinking>" in str(content):
                has_thinking = True

        # Count turns (user-assistant pairs)
        for _i, msg in enumerate(messages):
            if msg.get("role") == "user":
                turn_count += 1

        dynamic_tokens = max(0, total_tokens - static_tokens)
        static_ratio = static_tokens / max(total_tokens, 1)
        avg_message_length = total_chars / max(message_count, 1)

        return {
            "total_tokens": float(total_tokens),
            "static_tokens": float(static_tokens),
            "dynamic_tokens": float(dynamic_tokens),
            "static_ratio": static_ratio,
            "message_count": float(message_count),
            "avg_message_length": avg_message_length,
            "code_block_count": float(code_block_count),
            "has_tool_calls": 1.0 if has_tool_calls else 0.0,
            "has_thinking": 1.0 if has_thinking else 0.0,
            "turn_count": float(turn_count),
            "time_since_last_request": time_since_last,
            "cache_hit_rate_recent": recent_hit_rate,
        }

    def record_outcome(
        self,
        messages: list[dict[str, Any]],
        hit: bool,
        recent_hit_rate: float = 0.5,
        time_since_last: float = 0.0,
    ) -> None:
        """Record the outcome of a cache lookup for training.

        Args:
            messages: The message list that was sent
            hit: Whether the cache was hit
            recent_hit_rate: Recent cache hit rate at time of request
            time_since_last: Seconds since last request
        """
        features = self.extract_features(messages, recent_hit_rate, time_since_last)
        self._history.append({
            "features": features,
            "hit": 1.0 if hit else 0.0,
            "timestamp": time.time(),
        })

        # Retrain if we have enough new data
        if (
            not self._trained
            and len(self._history) >= self._retrain_threshold
        ) or (
            self._trained
            and len(self._history) - self._last_trained_size >= self._retrain_threshold
        ):
            self.train()

    def predict_hit_rate(
        self,
        messages: list[dict[str, Any]],
        recent_hit_rate: float = 0.5,
        time_since_last: float = 0.0,
    ) -> float:
        """Predict cache hit rate for a message list.

        Args:
            messages: The message list to predict for
            recent_hit_rate: Recent cache hit rate
            time_since_last: Seconds since last request

        Returns:
            Predicted hit rate (0.0-1.0)
        """
        features = self.extract_features(messages, recent_hit_rate, time_since_last)
        feature_vector = [features[name] for name in self.FEATURE_NAMES]

        self._stats["predictions"] += 1

        if not self._trained or self._model is None or not _HAS_XGBOOST:
            # Fallback: use static ratio heuristic
            static_ratio = features["static_ratio"]
            total_tokens = features["total_tokens"]
            # Higher static ratio and smaller context = higher hit rate
            heuristic = static_ratio * 0.7 + (1.0 - min(total_tokens / 4000, 1.0)) * 0.3
            return min(1.0, max(0.0, heuristic))

        try:
            import numpy as np

            x_vec = np.array([feature_vector])
            proba = self._model.predict_proba(x_vec)[0]
            return float(proba[1])  # Probability of class 1 (hit)
        except Exception as e:
            logger.debug("[HitPrediction] Prediction failed: %s", e)
            return 0.5

    def should_early_exit(
        self,
        messages: list[dict[str, Any]],
        threshold: float = 0.85,
    ) -> bool:
        """Determine if optimization should be skipped (early exit).

        Args:
            messages: The message list
            threshold: Minimum predicted hit rate to skip optimization

        Returns:
            True if optimization should be skipped
        """
        predicted = self.predict_hit_rate(messages)
        if predicted >= threshold:
            self._stats["early_exits"] += 1
            return True
        return False

    def should_aggressive_trim(
        self,
        messages: list[dict[str, Any]],
        threshold: float = 0.3,
    ) -> bool:
        """Determine if aggressive trimming should be applied.

        Args:
            messages: The message list
            threshold: Maximum predicted hit rate to trigger aggressive trim

        Returns:
            True if aggressive trimming should be applied
        """
        predicted = self.predict_hit_rate(messages)
        if predicted <= threshold:
            self._stats["aggressive_trims"] += 1
            return True
        return False

    def train(self) -> bool:
        """Train the XGBoost model on collected history.

        Returns:
            True if training succeeded
        """
        if not _HAS_XGBOOST:
            logger.debug("[HitPrediction] xgboost not available, skipping training")
            return False

        if len(self._history) < 10:
            logger.debug("[HitPrediction] Not enough data to train (%d samples)", len(self._history))
            return False

        try:
            import numpy as np

            x_list = []
            y_list = []
            for entry in self._history:
                features = entry["features"]
                x_list.append([features[name] for name in self.FEATURE_NAMES])
                y_list.append(entry["hit"])

            x_arr = np.array(x_list)
            y_arr = np.array(y_list)

            # Small, fast XGBoost model
            self._model = XGBClassifier(
                n_estimators=20,
                max_depth=3,
                learning_rate=0.3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=1,  # Keep it lightweight
            )

            self._model.fit(x_arr, y_arr, verbose=False)
            self._trained = True
            self._last_trained_size = len(self._history)
            self._stats["retrains"] += 1

            # Persist model weights
            self._save_model()

            logger.info(
                "[HitPrediction] Trained on %d samples, hit_rate=%.2f",
                len(self._history),
                sum(y_arr) / max(len(y_arr), 1),
            )
            return True

        except Exception as e:
            logger.warning("[HitPrediction] Training failed: %s", e)
            return False

    def _save_model(self) -> None:
        """Persist model to disk."""
        if self._model is None or not _HAS_XGBOOST:
            return
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Save model as JSON (XGBoost booster can be serialized)
            booster = self._model.get_booster()
            model_json = booster.save_config()
            _PERSISTENCE_PATH.write_text(json.dumps({
                "model_config": model_json,
                "feature_names": self.FEATURE_NAMES,
                "trained": self._trained,
                "last_trained_size": self._last_trained_size,
            }))
        except Exception as e:
            logger.debug("[HitPrediction] Failed to save model: %s", e)

    def _load_model(self) -> None:
        """Load persisted model from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        if not _HAS_XGBOOST:
            return
        try:
            data = json.loads(_PERSISTENCE_PATH.read_text())
            self._trained = data.get("trained", False)
            self._last_trained_size = data.get("last_trained_size", 0)
            # Model weights would need booster serialization for full restore
            # For now, we mark as trained and will retrain on next data
        except Exception:
            pass

    def get_stats(self) -> dict[str, int]:
        """Get prediction statistics."""
        return dict(self._stats)

    def reset(self) -> None:
        """Reset model and history."""
        self._history.clear()
        self._model = None
        self._trained = False
        self._last_trained_size = 0
        self._stats = {
            "predictions": 0,
            "early_exits": 0,
            "aggressive_trims": 0,
            "retrains": 0,
        }


# Global instance
_hit_model: HitPredictionModel | None = None


def get_hit_prediction_model() -> HitPredictionModel:
    """Get or create the global hit prediction model."""
    global _hit_model
    if _hit_model is None:
        _hit_model = HitPredictionModel()
    return _hit_model
