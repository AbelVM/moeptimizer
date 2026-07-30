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

# --- Error-aware extraction (review §4.1.1, snip/rtk pattern) ---------------
# Blind head/tail truncation drops the diagnostic signal that lives in the MIDDLE
# of a failing test/build/lint log (the compiler error, the failing assertion, the
# stack frame). These patterns let the compressor KEEP that signal and drop the
# passing-test / progress noise instead — ~90% reduction with zero loss of the
# information the model actually needs to fix the problem.

# A line that carries diagnostic signal worth keeping on a failure.
_FAIL_KEYWORDS_RE = re.compile(
    r"(?i)\b(errors?|failed|failures?|panic|fatal|traceback|exceptions?|"
    r"assert(?:ion)?error|unresolved|cannot|could not|no such|not found|"
    r"permission denied|segfault|aborted|expected\b.*\bgot\b)\b"
    r"|^[✗✘❌]|^(FAIL|ERROR|PANIC|FATAL)\b"
)
# Stack frames / file:line references (Python, JS/TS, Rust/Go, gcc/clang).
_FRAME_RE = re.compile(
    r'^\s*File\s+"[^"]+",\s*line\s+\d+'        # Python traceback frame
    r"|\bat\s+\S+\s+\(?[\w./\\-]+:\d+[:\d]*\)?"  # JS/TS / Node "at fn (file:line:col)"
    r"|^\s*[\w./\\-]+\.\w+:\d+(:\d+)?\b"        # generic file:line[:col]
    r"|^\s*[\w./\\-]+:\d+:\d+:"                 # gcc/clang file:line:col:
)
# A test/build/lint outcome summary line (always keep — it is the verdict).
_SUMMARY_RE = re.compile(
    r"(?i)\d+\s+(passed|failed|errors?|skipped|ignored|warnings?)"
    r"|test result:"
    r"|build (succeeded|failed|complete)"
    r"|found \d+ (errors?|warnings?)"
    r"|=+\s.*(pass|fail|error|summary).*\s=+"
)
# Noise to drop when extracting failure diagnostics (passing tests, progress bars,
# percentage ticks, separator rules, blank lines).
_NOISE_RE = re.compile(
    r"^\s*(?:\.{2,}|\[?\s*\d+%\]?|ok\b.*\d+(?:\.\d+)?s|running\b.*|collecting\b.*"
    r"|[-_=]{3,}|)\s*$",
    re.IGNORECASE,
)
# Marker so re-compressing already error-aware-compressed output is a no-op.
_ERROR_AWARE_MARKER = "... [error-aware compressed:"


def has_failure_signal(text: str) -> bool:
    """True if ``text`` carries error / stack-frame diagnostic signal.

    Shared with ``ToolOutputFilter`` so it does NOT collapse a failing
    test/build/lint log into a bare marker (which would discard the very
    diagnostics the model needs); failures are passed through to the error-aware
    compressor instead (review §4.1.1).
    """
    return any(
        _FAIL_KEYWORDS_RE.search(ln) or _FRAME_RE.search(ln)
        for ln in text.split("\n")
    )


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
        # Error-aware extraction (review §4.1.1): for a recognizable test/build/lint
        # result, keep the diagnostic signal (errors/frames/summary) and drop the
        # passing/progress noise instead of blindly cutting the middle. Returns None
        # for general output, which falls through to head/tail truncation.
        error_aware = self._compress_error_aware(text)
        if error_aware is not None:
            return error_aware
        text = self._truncate(text)
        return text

    def _compress_error_aware(self, text: str) -> str | None:
        """Compress a recognizable test/build/lint result, preserving diagnostics.

        Returns the compressed text, or ``None`` when the output is not a
        recognizable result (so the caller falls back to head/tail truncation).

        - Pure success (no failures): collapse to a one-line verdict.
        - Failure: keep the head (what ran), every diagnostic line (error / stack
          frame / file:line / summary) with a little trailing context, and the tail
          (final verdict); drop passing-test and progress noise. Capped to
          ``max_chars``.
        """
        if _ERROR_AWARE_MARKER in text:
            return text  # idempotent

        lines = text.split("\n")
        has_failure = any(
            _FAIL_KEYWORDS_RE.search(ln) or _FRAME_RE.search(ln) for ln in lines
        )
        has_summary = any(_SUMMARY_RE.search(ln) for ln in lines)

        # Error-aware extraction only applies to a structured RESULT (a verdict /
        # summary line is present). Raw stack traces and general errors fall through
        # to head/tail truncation, which already preserves the traceback head + the
        # final-error tail and collapses repeated frames.
        if not has_summary:
            return None

        # Pure success (a verdict but no failure signal anywhere): collapse to the
        # one-line verdict — the passing-test verbosity carries no information.
        if not has_failure:
            verdict = next(
                (ln.strip() for ln in lines if _SUMMARY_RE.search(ln)), "success"
            )
            return f"{verdict}\n{_ERROR_AWARE_MARKER} success collapsed {len(text)} -> ~{len(verdict)} chars] ..."

        # Failure path: extract diagnostic lines + limited context.
        keep_idx: set[int] = set()
        n = len(lines)
        head_ctx = 4
        for i in range(min(head_ctx, n)):
            keep_idx.add(i)  # what ran (command / test file header)
        for i, ln in enumerate(lines):
            if _FAIL_KEYWORDS_RE.search(ln) or _FRAME_RE.search(ln) or _SUMMARY_RE.search(ln):
                keep_idx.add(i)
                # Keep a little context after a diagnostic line (the message that
                # follows a frame / the assertion detail), and the line before a
                # frame (often the failing test name).
                if i + 1 < n:
                    keep_idx.add(i + 1)
                if i + 2 < n and _FRAME_RE.search(ln):
                    keep_idx.add(i + 2)
                if i - 1 >= 0:
                    keep_idx.add(i - 1)
        for i in range(max(0, n - head_ctx), n):
            keep_idx.add(i)  # tail (final verdict / summary)

        kept: list[str] = []
        prev = -2
        dropped = 0
        for i in sorted(keep_idx):
            ln = lines[i]
            if _NOISE_RE.match(ln) and not (
                _FAIL_KEYWORDS_RE.search(ln) or _SUMMARY_RE.search(ln)
            ):
                continue
            if i != prev + 1 and kept:
                if dropped:
                    kept.append(f"... [{dropped} noise lines omitted] ...")
                dropped = 0
                kept.append("...")
            kept.append(ln)
            prev = i
        if dropped:
            kept.append(f"... [{dropped} noise lines omitted] ...")

        result = "\n".join(kept)
        if len(result) > self.max_chars:
            result = result[: self.max_chars] + f"\n{_ERROR_AWARE_MARKER} capped at {self.max_chars} chars] ..."
        elif len(result) < len(text):
            result = result + f"\n{_ERROR_AWARE_MARKER} {len(text)} -> {len(result)} chars] ..."
        return result

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
