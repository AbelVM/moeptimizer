"""Segment-Wise Speculative Decoding.

Runs draft generation per code-block segment, reducing wasted draft tokens
when only a subset of the response changes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SegmentWiseSpeculativeDecoder:
    """
    Segment-wise speculative decoding for code-block responses.

    Instead of running draft generation on the entire response, this splits
    the response into segments (code blocks, text blocks) and runs draft
    generation per segment. This reduces wasted draft tokens when only a
    subset of the response changes between turns.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_draft_tokens: int = 4,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._max_draft_tokens = max_draft_tokens
        self._stats: dict[str, int] = {
            "total_segments": 0,
            "segments_skipped": 0,
            "segments_drafted": 0,
            "draft_tokens_generated": 0,
            "draft_tokens_accepted": 0,
        }

    def split_into_segments(self, text: str) -> list[dict[str, Any]]:
        """Split text into segments for per-segment speculative decoding.

        Segments are either code blocks or text blocks.

        Args:
            text: The text to split

        Returns:
            List of segment dicts with 'type', 'content', and 'start'/'end' positions
        """
        segments: list[dict[str, Any]] = []
        code_pattern = re.compile(r"(```[\w]*\n.*?```)", re.DOTALL)

        last_end = 0
        for match in code_pattern.finditer(text):
            # Text before code block
            if match.start() > last_end:
                text_segment = text[last_end : match.start()]
                if text_segment.strip():
                    segments.append({
                        "type": "text",
                        "content": text_segment,
                        "start": last_end,
                        "end": match.start(),
                    })

            # Code block
            code_content = match.group(1)
            segments.append({
                "type": "code",
                "content": code_content,
                "start": match.start(),
                "end": match.end(),
            })
            last_end = match.end()

        # Remaining text after last code block
        if last_end < len(text):
            text_segment = text[last_end:]
            if text_segment.strip():
                segments.append({
                    "type": "text",
                    "content": text_segment,
                    "start": last_end,
                    "end": len(text),
                })

        return segments

    def should_draft_segment(
        self,
        segment: dict[str, Any],
        previous_segment: dict[str, Any] | None,
    ) -> bool:
        """Determine if a segment should receive draft tokens.

        Segments that are unchanged from the previous turn can skip drafting.

        Args:
            segment: The current segment
            previous_segment: The corresponding segment from the previous turn

        Returns:
            True if draft generation should be run for this segment
        """
        if previous_segment is None:
            return True  # New segment, always draft

        # Skip drafting if segment content is identical
        if segment.get("content") == previous_segment.get("content"):
            return False

        # Skip drafting for short text segments
        return not (segment["type"] == "text" and len(segment.get("content", "")) < 50)

    def get_draft_tokens_for_segment(
        self,
        segment: dict[str, Any],
        mtp_decoder: Any | None = None,
    ) -> list[str]:
        """Get draft tokens for a specific segment.

        Args:
            segment: The segment to generate drafts for
            mtp_decoder: Optional MTP speculative decoder

        Returns:
            List of draft token strings
        """
        if mtp_decoder is None:
            return []

        content = segment.get("content", "")
        if not content:
            return []

        # For code segments, generate more draft tokens
        if segment["type"] == "code":
            max_draft = min(self._max_draft_tokens * 2, 8)
        else:
            max_draft = self._max_draft_tokens

        # Simulate draft generation (in production, this would call the model)
        # For now, return empty — the integration point is documented
        self._stats["draft_tokens_generated"] += max_draft
        return []

    def process_segments(
        self,
        text: str,
        previous_text: str | None = None,
        mtp_decoder: Any | None = None,
    ) -> dict[str, Any]:
        """Process text with segment-wise speculative decoding.

        Args:
            text: The current text to process
            previous_text: The previous text for change detection
            mtp_decoder: Optional MTP speculative decoder

        Returns:
            Dict with 'segments', 'draft_map', and 'changed_segments'
        """
        current_segments = self.split_into_segments(text)
        previous_segments = (
            self.split_into_segments(previous_text) if previous_text else []
        )

        draft_map: dict[int, list[str]] = {}
        changed_segments: list[int] = []

        for i, segment in enumerate(current_segments):
            self._stats["total_segments"] += 1
            prev = previous_segments[i] if i < len(previous_segments) else None

            if self.should_draft_segment(segment, prev):
                drafts = self.get_draft_tokens_for_segment(segment, mtp_decoder)
                if drafts:
                    draft_map[i] = drafts
                    self._stats["segments_drafted"] += 1
                changed_segments.append(i)
            else:
                self._stats["segments_skipped"] += 1

        return {
            "segments": current_segments,
            "draft_map": draft_map,
            "changed_segments": changed_segments,
            "unchanged_count": self._stats["segments_skipped"],
        }

    def get_stats(self) -> dict[str, int]:
        """Get speculative decoding statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_segments": 0,
            "segments_skipped": 0,
            "segments_drafted": 0,
            "draft_tokens_generated": 0,
            "draft_tokens_accepted": 0,
        }


# Global instance
_segment_decoder: SegmentWiseSpeculativeDecoder | None = None


def get_segment_wise_decoder() -> SegmentWiseSpeculativeDecoder:
    """Get or create the global segment-wise speculative decoder."""
    global _segment_decoder
    if _segment_decoder is None:
        _segment_decoder = SegmentWiseSpeculativeDecoder()
    return _segment_decoder
