"""ThinkingPreserver — Pass-through for front-loading eviction.

Since the compactor now uses pure eviction (dropping entire turns), there is
no need to compress reasoning content. The ThinkingPreserver preserves all
messages as-is, ensuring the MTP heads see exactly the token sequences they
were trained on.

The `protect_recent` parameter is retained for API compatibility but has no
functional effect — all messages are preserved regardless of recency.
"""

from __future__ import annotations

from typing import Any


class ThinkingPreserver:
    """
    Pass-through reasoning preserver for front-loading eviction.

    Since eviction drops entire turns (no summarization), reasoning tags
    are never modified. This preserves the exact token sequences the model
    was trained on, preventing MTP head disruption and Vulkan prefills.

    The `protect_recent` parameter is retained for API compatibility.
    """

    def __init__(self, protect_recent: int | None = None) -> None:
        # Parameter retained for API compatibility; no functional effect
        # since eviction drops whole turns rather than compressing content.
        pass

    def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Return messages unchanged.

        With front-loading eviction, reasoning content is never compressed.
        Old turns are dropped entirely (not summarized), so all preserved
        messages retain their original token sequences.
        """
        return [dict(msg) for msg in messages]
