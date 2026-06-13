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
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._max_size = max_size
        self._stats: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
        self._pattern_cache: OrderedDict[str, str] = OrderedDict()

    def get(self, token_pattern: str) -> tuple[int, ...] | None:
        """Get cached expert mask for a token pattern."""
        if token_pattern in self._cache:
            self._stats["hits"] += 1
            self._cache.move_to_end(token_pattern)
            return self._cache[token_pattern]
        self._stats["misses"] += 1
        return None

    def put(self, token_pattern: str, expert_mask: tuple[int, ...]) -> None:
        """Cache an expert routing decision."""
        if token_pattern in self._cache:
            self._cache.move_to_end(token_pattern)
        self._cache[token_pattern] = expert_mask
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    def get_or_compute(
        self,
        token_pattern: str,
        compute_fn: Any,
    ) -> tuple[int, ...]:
        """Get cached expert mask or compute and cache it."""
        cached = self.get(token_pattern)
        if cached is not None:
            return cached
        result = compute_fn()
        self.put(token_pattern, result)
        return result

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

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
            cached = self.get(pattern)
            if cached is not None:
                return cached

        return None

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
        """
        # Extract common patterns from static content
        patterns = self._extract_static_patterns(static_content)

        # For each pattern, we'd normally get the expert mask from the model
        # For now, we just record the patterns
        for pattern in patterns:
            if pattern not in self._cache:
                # Placeholder - would be filled by actual model routing
                self._cache[pattern] = tuple(range(64))  # Default 64 experts

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