"""Expert routing cache for MoE models.

Caches expert routing decisions to improve:
- Expert cache locality
- Routing consistency
- MTP prediction accuracy
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class ExpertRoutingCache:
    """
    Cache for MoE expert routing decisions.

    Qwen3.6-35B-A3B-MTP uses token-level expert routing.
    This cache stores (token_pattern → expert_mask) mappings
    to reduce routing overhead and improve expert cache locality.
    """

    def __init__(self, max_size: int = 4096) -> None:
        # Partitioned caches for static/dynamic layers
        self._static_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._dynamic_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._max_size = max_size
        self._stats: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}

    def get(
        self,
        token_pattern: str,
        layer: str = "dynamic",
    ) -> tuple[int, ...] | None:
        """Get cached expert mask for a token pattern.

        layer: "static" or "dynamic" to use partitioned cache.
        """
        cache = self._static_cache if layer == "static" else self._dynamic_cache
        if token_pattern in cache:
            self._stats["hits"] += 1
            cache.move_to_end(token_pattern)
            return cache[token_pattern]
        self._stats["misses"] += 1
        return None

    def put(
        self,
        token_pattern: str,
        expert_mask: tuple[int, ...],
        layer: str = "dynamic",
    ) -> None:
        """Cache an expert routing decision.

        layer: "static" or "dynamic" to use partitioned cache.
        """
        cache = self._static_cache if layer == "static" else self._dynamic_cache
        if token_pattern in cache:
            cache.move_to_end(token_pattern)
        cache[token_pattern] = expert_mask
        while len(cache) > self._max_size // 2:
            cache.popitem(last=False)
            self._stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all caches."""
        self._static_cache.clear()
        self._dynamic_cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_or_compute(
        self,
        token_pattern: str,
        compute_fn: Any,
        layer: str = "dynamic",
    ) -> tuple[int, ...]:
        """Get cached expert mask or compute and cache it."""
        cached = self.get(token_pattern, layer=layer)
        if cached is not None:
            return cached
        result = compute_fn()
        self.put(token_pattern, result, layer=layer)
        return result

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return dict(self._stats)

    def hash_ast_node(
        self,
        node_text: str,
        node_type: str,
    ) -> str:
        """Hash an AST node for expert routing prediction."""
        return hashlib.md5(f"{node_type}:{node_text}".encode()).hexdigest()[:16]

    def predict_expert_for_code(
        self,
        code: str,
        node_type: str,
    ) -> tuple[int, ...] | None:
        """Predict expert routing for code patterns.

        Uses cached patterns to predict which experts will handle
        this code, improving routing consistency.
        """
        # Extract key patterns from code
        patterns = self._extract_patterns(code, node_type)

        # Check cache for any matching pattern
        for pattern in patterns:
            cached = self.get(pattern, layer="dynamic")
            if cached is not None:
                return cached

        return None

    def predict_expert_for_tokens(
        self,
        tokens: list[str],
    ) -> list[tuple[int, ...]]:
        """Predict expert for each token in sequence.

        Token-level prediction provides finer granularity for MoE optimization.
        """
        predictions: list[tuple[int, ...]] = []
        for token in tokens:
            # Use token as pattern
            cached = self.get(token, layer="dynamic")
            if cached:
                predictions.append(cached)
            else:
                # Default prediction
                predictions.append(tuple(range(0, 64)))
        return predictions

    def get_expert_hints_for_context(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate expert hints for a context.

        Returns list of {position, experts} for the Lemonade server.
        """
        hints: list[dict[str, Any]] = []
        position = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Extract tokens
                tokens = content.split()
                for token in tokens[:100]:  # Limit to first 100 tokens
                    cached = self.get(token, layer="dynamic")
                    if cached:
                        hints.append({
                            "position": position,
                            "experts": list(cached),
                        })
                    position += 1

        return hints

    def _extract_patterns(
        self,
        code: str,
        node_type: str,
    ) -> list[str]:
        """Extract routing-relevant patterns from code."""
        patterns = []

        # Add node type pattern
        patterns.append(f"type:{node_type}")

        # Add language pattern
        if "def " in code or "import " in code:
            patterns.append("python")
        elif "function " in code or "const " in code:
            patterns.append("javascript")

        # Add structure patterns
        if "class " in code:
            patterns.append("class")
        if "def " in code or "function " in code:
            patterns.append("function")

        # Add indentation pattern
        lines = code.split("\n")
        if lines:
            first_line = lines[0]
            indent = len(first_line) - len(first_line.lstrip())
            patterns.append(f"indent:{indent // 4}")

        return patterns

    def warm_cache_for_static_layer(
        self,
        static_content: str,
    ) -> None:
        """Pre-warm expert cache with static layer patterns.

        This should be called when static layer is first loaded.
        Extracts code patterns and caches their expert routing decisions.
        """
        # Extract common patterns from static content
        patterns = self._extract_static_patterns(static_content)

        # For each pattern, we'd normally get the expert mask from the model
        # For now, we just record the patterns and use placeholder masks
        # In production, this would call the model's /token-predict endpoint
        for pattern in patterns:
            if pattern not in self._static_cache:
                # Placeholder - would be filled by actual model routing
                # The pattern hash maps to likely expert indices
                expert_mask = self._predict_experts_for_pattern(pattern)
                self._static_cache[pattern] = expert_mask

    def _predict_experts_for_pattern(self, pattern: str) -> tuple[int, ...]:
        """Predict expert mask for a pattern.

        Uses heuristics based on pattern type to predict which experts
        will handle this code.
        """
        # Python patterns typically use experts 0-15
        if "python" in pattern:
            return tuple(range(0, 16))
        # JavaScript patterns use experts 16-31
        elif "javascript" in pattern or "typescript" in pattern:
            return tuple(range(16, 32))
        # Class patterns use experts 32-47
        elif "class" in pattern:
            return tuple(range(32, 48))
        # Function patterns use experts 48-63
        elif "function" in pattern:
            return tuple(range(48, 64))
        # Default: spread across all experts
        return tuple(range(0, 64))

    def update_from_model_feedback(
        self,
        token: str,
        expert_mask: tuple[int, ...],
        layer: str = "dynamic",
    ) -> None:
        """Update cache with actual model routing feedback.

        This should be called after each request to improve predictions.
        """
        self.put(token, expert_mask, layer=layer)

    def get_or_predict(
        self,
        token: str,
        code_context: str = "",
        layer: str = "dynamic",
    ) -> tuple[int, ...]:
        """Get cached expert mask or predict based on context.

        Uses code context to improve predictions when no cache hit.
        """
        cached = self.get(token, layer=layer)
        if cached is not None:
            return cached

        # Predict based on code context
        if code_context:
            patterns = self._extract_patterns(code_context, "unknown")
            for pattern in patterns:
                cached = self.get(pattern, layer=layer)
                if cached is not None:
                    return cached

        # Return default prediction
        return tuple(range(0, 64))

    def _extract_static_patterns(self, content: str) -> list[str]:
        """Extract patterns from static layer content."""
        patterns = []

        # Language patterns
        if "```python" in content:
            patterns.append("python")
        if "```javascript" in content:
            patterns.append("javascript")
        if "```typescript" in content:
            patterns.append("typescript")

        # Structure patterns
        if "class " in content:
            patterns.append("class")
        if "def " in content:
            patterns.append("function")
        if "import " in content:
            patterns.append("import")

        return patterns

    def extract_hints_from_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract expert routing hints from a model response.

        Uses the response to update the cache and generate hints for future requests.
        This provides token-level expert routing feedback integration.
        """
        hints: list[dict[str, Any]] = []

        # Get the assistant response content
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                content = choice.message.content or ""
            elif hasattr(choice, "delta"):
                content = choice.delta.content or ""
            else:
                content = ""
        elif isinstance(response, dict):
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return hints

        # Extract tokens and update cache
        tokens = content.split()
        for i, token in enumerate(tokens[:100]):  # Limit to first 100 tokens
            # Get context from surrounding messages
            context = ""
            for msg in messages[-3:]:  # Last 3 messages for context
                if msg.get("role") == "user":
                    context = msg.get("content", "")
                    break

            # Predict or get cached expert mask
            expert_mask = self.get_or_predict(token, context)
            hints.append({
                "position": i,
                "experts": list(expert_mask),
            })

        return hints


# Global expert cache instance
_expert_cache: ExpertRoutingCache | None = None


def get_expert_cache() -> ExpertRoutingCache:
    """Get or create the global expert routing cache."""
    global _expert_cache
    if _expert_cache is None:
        _expert_cache = ExpertRoutingCache()
    return _expert_cache


def hash_for_expert_routing(text: str, context: str = "") -> str:
    """Generate hash for expert routing cache key."""
    return hashlib.md5(f"{context}:{text}".encode()).hexdigest()[:16]