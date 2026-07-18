"""Tool output streaming for large outputs.

Streams large tool outputs in MTP-aware chunks to avoid context bloat
while maintaining prediction quality.
"""

from __future__ import annotations

import re


class ToolOutputStreamer:
    """
    Streams large tool outputs in MTP-aware chunks.

    Large tool outputs (terminal logs, file dumps) are split into
    manageable chunks that preserve MTP prediction patterns.
    """

    CHUNK_SIZE = 1024  # Tokens per chunk
    OVERLAP_SIZE = 128  # Overlap for MTP state continuity

    def __init__(self) -> None:
        self._chunk_cache: dict[str, list[str]] = {}

    def should_stream(
        self,
        content: str,
    ) -> bool:
        """Determine if content should be streamed.

        Large outputs (>500 lines or >4K chars) are candidates for streaming.
        """
        if not content:
            return False

        # Check line count
        lines = content.split("\n")
        if len(lines) > 500:
            return True

        # Check character count
        return len(content) > 4000

    def stream_output(
        self,
        content: str,
        tool_name: str,
    ) -> list[str]:
        """Split tool output into streamable chunks.

        Returns list of content strings (not full messages) to preserve role.
        The caller is responsible for wrapping in appropriate message structure.
        """
        if not self.should_stream(content):
            return [content]

        # Split into chunks
        chunks = self._split_into_chunks(content)

        # Return just the content strings
        return chunks

    def _split_into_chunks(
        self,
        content: str,
    ) -> list[str]:
        """Split content into MTP-friendly chunks."""
        # For code, split at function/class boundaries
        if "```" in content:
            return self._split_code_content(content)

        # For logs, split at line boundaries
        lines = content.split("\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for line in lines:
            if current_size + len(line) > self.CHUNK_SIZE and current_chunk:
                chunks.append("\n".join(current_chunk))
                # Keep overlap
                overlap_start = max(0, len(current_chunk) - 10)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(chunk_line) for chunk_line in current_chunk)
            current_chunk.append(line)
            current_size += len(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _split_code_content(
        self,
        content: str,
    ) -> list[str]:
        """Split code content at block boundaries."""
        # Extract code blocks
        code_pattern = re.compile(
            r"(```[\w]*\n.*?```)",
            re.DOTALL,
        )

        # Split into code and non-code sections
        parts = []
        last_end = 0

        for match in code_pattern.finditer(content):
            if match.start() > last_end:
                parts.append(content[last_end : match.start()])
            parts.append(match.group(1))
            last_end = match.end()

        if last_end < len(content):
            parts.append(content[last_end:])

        # Group into chunks
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for part in parts:
            if current_size + len(part) > self.CHUNK_SIZE and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(part)
            current_size += len(part)

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


def get_tool_streamer() -> ToolOutputStreamer:
    """Get a tool output streamer instance."""
    return ToolOutputStreamer()
