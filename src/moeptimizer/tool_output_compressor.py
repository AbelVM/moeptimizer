"""Tool/assistant output boundary compression (headroom/snip-style).

The biggest agentic-context bloat is tool I/O: terminal logs, file dumps, RAG
blobs, long assistant outputs. This module applies cheap, lossless-ish transforms
to those outputs *at the boundary* — before they enter the stable leading prefix —
so the backend's native prefix cache stays valid (the compressed form is frozen
into the prefix on first appearance) while token count drops.

Transforms (all conservative; never drop the head or unique lines):
- strip ANSI escape sequences / carriage returns
- collapse 3+ identical consecutive lines to a single line + a repeat count
- collapse repeated stack-frame blocks (``File "...", line N, in ...``) that
  recur verbatim
- truncate oversized outputs, keeping the head and tail with a marker
- keep code signatures (``def``/``class``/``async def`` lines) intact

This is intentionally NOT semantic summarization: it does not rewrite meaning,
so quality (semantic similarity vs direct) is preserved far better than folding
turns into a summary. See review03.md §3 / §5.1.
"""

from __future__ import annotations

import re
from typing import Any

# ANSI escape sequences (color codes, cursor moves, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CR_RE = re.compile(r"\r\n|\r")

# A repeated stack frame line, e.g. '  File "x.py", line 12, in foo'
_STACK_FRAME_RE = re.compile(r'^\s*File\s+"[^"]+",\s*line\s+\d+,\s*in\s+\S+')


class ToolOutputCompressor:
    """Boundary-compress large tool/assistant outputs with cheap transforms."""

    def __init__(self, max_chars: int = 4000, keep_head_ratio: float = 0.6) -> None:
        self.max_chars = max_chars if max_chars > 0 else 4000
        self.keep_head_ratio = max(0.1, min(0.9, keep_head_ratio))

    def should_compress(self, content: str) -> bool:
        """True if the content is large enough to be worth compressing."""
        return bool(content) and len(content) > self.max_chars

    def compress(self, content: str) -> str:
        """Return a compressed form of ``content`` (idempotent on small input)."""
        if not content or not self.should_compress(content):
            return content

        text = self._strip_ansi(content)
        # Collapse structure-preserving transforms FIRST (while the repeated
        # structure is intact), then truncate last. Stack-frame collapse runs
        # before repeated-line collapse so its omission marker is not re-folded.
        # This order is also what makes compress() idempotent.
        text = self._collapse_repeated_stack_frames(text)
        text = self._collapse_repeated_lines(text)
        text = self._truncate(text)
        return text

    # --- internals ---------------------------------------------------------

    def _strip_ansi(self, text: str) -> str:
        text = _CR_RE.sub("\n", text)
        return _ANSI_RE.sub("", text)

    def _collapse_repeated_lines(self, text: str) -> str:
        """Collapse 3+ identical consecutive lines into one + a repeat count."""
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        n = len(lines)
        while i < n:
            j = i + 1
            while j < n and lines[j] == lines[i] and lines[i] != "":
                j += 1
            run = j - i
            if run >= 3:
                out.append(lines[i])
                out.append(f"... (repeated {run} times) ...")
            else:
                out.extend(lines[i:j])
            i = j
        return "\n".join(out)

    def _collapse_repeated_stack_frames(self, text: str) -> str:
        """Collapse repeated stack-frame blocks (recurring verbatim frames).

        The first occurrence of a contiguous stack-frame block is kept in full;
        any later verbatim repeat of that same block is replaced by a short
        marker. This shrinks retry-loop tracebacks that dump the same frames
        again and again, while preserving the traceback the first time it is
        seen. Idempotent: an already-emitted marker is not itself a frame block.
        """
        lines = text.split("\n")
        out: list[str] = []
        seen: set[str] = set()
        i = 0
        n = len(lines)
        while i < n:
            if _STACK_FRAME_RE.match(lines[i]):
                # Collect a contiguous run of stack-frame lines.
                j = i
                while j < n and _STACK_FRAME_RE.match(lines[j]):
                    j += 1
                block = lines[i:j]
                key = "\n".join(block)
                if key in seen:
                    out.append(
                        f"... (stack frame block repeated; omitted {len(block)} lines) ..."
                    )
                else:
                    seen.add(key)
                    out.extend(block)
                i = j
            else:
                out.append(lines[i])
                i += 1
        return "\n".join(out)

    _TRUNC_MARKER = "... [tool output truncated:"

    def _truncate(self, text: str) -> str:
        """Truncate oversized text, keeping head + tail with a marker.

        Idempotent: if the text already contains a truncation marker it is
        returned unchanged, so re-running compress() on frozen output is safe.
        """
        if len(text) <= self.max_chars:
            return text
        if self._TRUNC_MARKER in text:
            return text
        head_chars = int(self.max_chars * self.keep_head_ratio)
        tail_chars = self.max_chars - head_chars
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""
        return (
            f"{head}\n"
            f"... [tool output truncated: {len(text) - self.max_chars} chars omitted] ...\n"
            f"{tail}"
        )


def compress_tool_messages(
    messages: list[dict[str, Any]],
    compressor: ToolOutputCompressor,
    roles: tuple[str, ...] = ("tool", "assistant"),
) -> list[dict[str, Any]]:
    """Return ``messages`` with large tool/assistant outputs boundary-compressed.

    Only string ``content`` is touched; structured (list) content is left as-is.
    Idempotent: already-compressed (small) outputs are returned unchanged, so
    re-running on a frozen prefix is safe.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in roles and isinstance(content, str) and compressor.should_compress(content):
            new_msg = dict(msg)
            new_msg["content"] = compressor.compress(content)
            out.append(new_msg)
        else:
            out.append(msg)
    return out
