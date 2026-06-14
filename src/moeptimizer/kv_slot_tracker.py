"""KV-slot tracking for explicit cache control.

Tracks which KV-cache slots are occupied by which context and provides
cache control hints for llama.cpp to manage memory efficiently.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KVSplit:
    """Represents a range of KV-cache slots occupied by a context segment."""

    context_hash: str
    start_slot: int
    end_slot: int
    content_preview: str = ""
    is_static: bool = False
    is_overlap: bool = False


@dataclass
class KVSlotMap:
    """Maps context to cache slots for a single request."""

    slots: list[KVSplit] = field(default_factory=list)
    total_slots: int = 0

    def add_split(
        self,
        content: str,
        start_slot: int,
        is_static: bool = False,
    ) -> KVSplit:
        """Add a content split to the slot map."""
        # Estimate slots: ~128 tokens per slot for Qwen3.6-35B-A3B-MTP
        estimated_tokens = len(content) // 4
        slot_count = max(1, estimated_tokens // 128)
        end_slot = start_slot + slot_count

        split = KVSplit(
            context_hash=hashlib.md5(content.encode()).hexdigest()[:32],
            start_slot=start_slot,
            end_slot=end_slot,
            content_preview=content[:100],
            is_static=is_static,
        )
        self.slots.append(split)
        self.total_slots = max(self.total_slots, end_slot)
        return split

    def get_eviction_hints(self) -> list[dict[str, Any]]:
        """Get cache control hints for slots to evict.

        Returns list of {slot_start, slot_end} for content that should be evicted.
        """
        # Evict non-static, non-overlap slots first
        hints = []
        for split in self.slots:
            if not split.is_static and not split.is_overlap:
                hints.append({
                    "slot_start": split.start_slot,
                    "slot_end": split.end_slot,
                })
        return hints

    def get_preserve_hints(self) -> list[dict[str, Any]]:
        """Get cache control hints for slots to preserve.

        Returns list of {slot_start, slot_end} for content that should be kept.
        """
        hints = []
        for split in self.slots:
            if split.is_static or split.is_overlap:
                hints.append({
                    "slot_start": split.start_slot,
                    "slot_end": split.end_slot,
                })
        return hints


class KVSlotTracker:
    """
    Tracks KV-cache slot occupancy across requests.

    When context is evicted and restored, this module provides hints to
    llama.cpp's cache_control API to explicitly manage cache slots.
    """

    def __init__(self, block_size: int = 128) -> None:
        self._block_size = block_size
        self._session_maps: dict[str, KVSlotMap] = {}
        self._global_hot_slots: set[int] = set()
        self._stats: dict[str, int] = {
            "slots_tracked": 0,
            "slots_evicted": 0,
            "slots_preserved": 0,
        }

    def build_slot_map(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> KVSlotMap:
        """Build a slot map for the given messages.

        Assigns slot ranges to each message based on estimated token count.
        Static layer (system + first user) gets special marking.
        """
        slot_map = KVSlotMap()
        current_slot = 0

        # Find static layer end
        static_end = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                static_end = i + 1
            elif msg.get("role") == "user" and static_end > 0:
                static_end = i + 1
                break
            elif msg.get("role") == "user" and static_end == 0:
                static_end = i + 1
                break

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                is_static = i < static_end
                slot_map.add_split(content, current_slot, is_static=is_static)
                # Update current slot
                estimated_tokens = len(content) // 4
                current_slot += max(1, estimated_tokens // self._block_size)

        if session_id:
            self._session_maps[session_id] = slot_map

        self._stats["slots_tracked"] += slot_map.total_slots
        return slot_map

    def get_cache_control_hints(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Get cache control hints for the backend.

        Returns a dict suitable for passing to llama.cpp's cache_control.
        """
        slot_map = self.build_slot_map(messages, session_id)

        return {
            "cache_control": {
                "preserve_slots": slot_map.get_preserve_hints(),
                "evict_slots": slot_map.get_eviction_hints(),
            },
        }

    def mark_hot_slots(self, slot_map: KVSlotMap) -> None:
        """Mark slots as hot (frequently accessed) for priority preservation."""
        for split in slot_map.slots:
            if split.is_static:
                for slot in range(split.start_slot, split.end_slot):
                    self._global_hot_slots.add(slot)

    def get_hot_slot_ratio(self) -> float:
        """Get the ratio of hot slots to total tracked slots."""
        if self._stats["slots_tracked"] == 0:
            return 0.0
        return len(self._global_hot_slots) / self._stats["slots_tracked"]

    def get_stats(self) -> dict[str, int]:
        """Get slot tracking statistics."""
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all slot maps."""
        self._session_maps.clear()
        self._global_hot_slots.clear()
        self._stats = {
            "slots_tracked": 0,
            "slots_evicted": 0,
            "slots_preserved": 0,
        }


# Global slot tracker instance
_slot_tracker: KVSlotTracker | None = None


def get_kv_slot_tracker() -> KVSlotTracker:
    """Get or create the global KV slot tracker."""
    global _slot_tracker
    if _slot_tracker is None:
        _slot_tracker = KVSlotTracker()
    return _slot_tracker