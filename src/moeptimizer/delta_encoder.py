"""Delta-Encoding of Code for context compression.

Stores only diffs between successive code snapshots; reconstructs full code
when needed, cutting context size for repeated code.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Persistence path
_PERSISTENCE_PATH = Path.home() / ".moeptimizer" / "code_deltas.json"


class CodeDeltaEncoder:
    """
    Delta-encodes code snapshots to reduce context size.

    Instead of storing full code on each turn, stores only the diff
    between successive snapshots. Reconstructs full code when needed.
    This dramatically reduces context size for repeated code patterns.
    """

    def __init__(self, max_snapshots: int = 100) -> None:
        self._snapshots: OrderedDict[str, str] = OrderedDict()
        self._deltas: OrderedDict[str, str] = OrderedDict()
        self._max_snapshots = max_snapshots
        # file_path -> most-recent snapshot key, so _find_previous_key can locate
        # the prior version of a file (previously a no-op, so every snapshot
        # stored full content and deltas were never produced).
        self._file_keys: dict[str, str] = {}
        # file_path -> content stored *before* the most recent store_snapshot for
        # that path. Lets callers retrieve the delta of the latest version vs the
        # immediately preceding one (used for re-read diff injection, P2.2).
        self._prev_content: dict[str, str] = {}
        self._stats: dict[str, int] = {
            "snapshots_stored": 0,
            "deltas_stored": 0,
            "reconstructions": 0,
            "bytes_saved": 0,
        }

    def _make_key(self, file_path: str, content_hash: str) -> str:
        """Generate a stable key for a code snapshot."""
        return hashlib.md5(f"{file_path}:{content_hash}".encode()).hexdigest()[:32]

    def store_snapshot(
        self,
        file_path: str,
        content: str,
    ) -> str:
        """Store a code snapshot, computing delta from previous version.

        Args:
            file_path: The file path (used as identifier)
            content: The full code content

        Returns:
            Snapshot key
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        key = self._make_key(file_path, content_hash)

        # Check if we have a previous version
        prev_key = self._find_previous_key(file_path)
        prev_content = self._snapshots.get(prev_key, "") if prev_key else ""

        if prev_content and prev_content != content:
            # Compute and store delta
            delta = self._compute_delta(prev_content, content)
            self._deltas[key] = delta
            self._stats["deltas_stored"] += 1
            saved = len(content) - len(delta)
            self._stats["bytes_saved"] += max(0, saved)
        elif not prev_content:
            # First snapshot — store full content
            self._deltas[key] = content

        # Store snapshot
        self._snapshots[key] = content
        self._snapshots.move_to_end(key)

        # Track file_path -> most-recent key so _find_previous_key can locate the
        # prior version of the same file (previously a no-op, so deltas were
        # never produced and every snapshot stored full content).
        self._file_keys[file_path] = key
        # Remember the content that preceded this store for the same path, so a
        # later caller can retrieve the delta of this version vs the prior one.
        self._prev_content[file_path] = prev_content

        # Evict oldest if over limit
        while len(self._snapshots) > self._max_snapshots:
            old_key, _ = self._snapshots.popitem(last=False)
            self._deltas.pop(old_key, None)
            # Drop any file_path mapping that pointed at the evicted key.
            for fp, k in list(self._file_keys.items()):
                if k == old_key:
                    self._file_keys.pop(fp, None)
            self._prev_content.pop(file_path, None)

        self._stats["snapshots_stored"] += 1
        return key

    def _find_previous_key(self, file_path: str) -> str | None:
        """Find the most recent snapshot key for a file path."""
        return self._file_keys.get(file_path)

    def get_previous_content(self, file_path: str) -> str | None:
        """Return the content stored *before* the most recent store_snapshot for
        ``file_path`` (None if this is the first version seen for that path)."""
        prev = self._prev_content.get(file_path)
        return prev if prev else None

    def get_delta(self, key: str) -> str:
        """Return the stored delta for ``key`` (empty string if none)."""
        return self._deltas.get(key, "")

    def get_delta_vs_previous(self, file_path: str, content: str) -> str:
        """Return the diff of ``content`` against the previously stored version
        of ``file_path`` (empty string if this is the first version or unchanged).

        Used by the optimizer to inject a compact diff when a file is re-read
        after an edit (P2.2). The caller is responsible for confirming the prior
        version is already present in the model context before injecting.
        """
        prev = self._prev_content.get(file_path, "")
        if not prev or prev == content:
            return ""
        return self._compute_delta(prev, content)

    def _compute_delta(self, old: str, new: str) -> str:
        """Compute a compact diff between old and new content.

        Uses unified diff format, compressed to store only the changed lines
        (no unchanged context), so the delta is smaller than the full content
        for any non-trivial edit.
        """
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm="",
            n=0,  # No context lines — keep only added/removed lines.
        ))

        # Keep only added/removed lines; drop the ---/+++/@@ header lines and
        # any unchanged context lines (space-prefixed).
        changed_lines = [
            line
            for line in diff
            if line and not line.startswith(("---", "+++", "@@", " "))
        ]

        if not changed_lines:
            return ""

        # Compress: store as a compact patch
        return "\n".join(changed_lines)

    def reconstruct(self, key: str) -> str | None:
        """Reconstruct full code from a snapshot key.

        Args:
            key: The snapshot key

        Returns:
            Full code content or None if not found
        """
        if key in self._snapshots:
            self._stats["reconstructions"] += 1
            return self._snapshots[key]

        # Try to reconstruct from delta
        if key in self._deltas:
            delta = self._deltas[key]
            # If delta looks like full content (no diff markers), return it
            if not delta.startswith(("-", "+", "@")):
                self._stats["reconstructions"] += 1
                return delta

        return None

    def get_delta_size(self, key: str) -> int:
        """Get the size of the stored delta for a snapshot."""
        delta = self._deltas.get(key, "")
        return len(delta)

    def get_full_size(self, key: str) -> int:
        """Get the size of the full snapshot."""
        content = self._snapshots.get(key, "")
        return len(content)

    def get_compression_ratio(self, key: str) -> float:
        """Get the compression ratio for a snapshot (delta/full)."""
        full = self.get_full_size(key)
        if full == 0:
            return 0.0
        return round(self.get_delta_size(key) / full, 3)

    def get_or_create(
        self,
        file_path: str,
        content: str,
    ) -> tuple[str, bool]:
        """Get existing snapshot or create new one.

        Args:
            file_path: The file path
            content: The code content

        Returns:
            Tuple of (key, is_new)
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        key = self._make_key(file_path, content_hash)

        if key in self._snapshots:
            return key, False

        self.store_snapshot(file_path, content)
        return key, True

    def get_stats(self) -> dict[str, Any]:
        """Get delta encoding statistics."""
        total_full = sum(self.get_full_size(k) for k in self._snapshots)
        total_delta = sum(self.get_delta_size(k) for k in self._deltas)
        return {
            **self._stats,
            "total_snapshots": len(self._snapshots),
            "total_deltas": len(self._deltas),
            "total_full_bytes": total_full,
            "total_delta_bytes": total_delta,
            "overall_compression": round(total_delta / max(total_full, 1), 3),
        }

    def clear(self) -> None:
        """Clear all snapshots and deltas."""
        self._snapshots.clear()
        self._deltas.clear()
        self._stats = {
            "snapshots_stored": 0,
            "deltas_stored": 0,
            "reconstructions": 0,
            "bytes_saved": 0,
        }

    def save_to_disk(self) -> None:
        """Persist snapshots and deltas to disk."""
        try:
            _PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "snapshots": dict(self._snapshots),
                "deltas": dict(self._deltas),
                "stats": self._stats,
            }
            _PERSISTENCE_PATH.write_text(json.dumps(data))
        except Exception as e:
            logger.warning("[DeltaEncoder] Failed to save: %s", e)

    def load_from_disk(self) -> None:
        """Load snapshots and deltas from disk."""
        if not _PERSISTENCE_PATH.exists():
            return
        try:
            data = json.loads(_PERSISTENCE_PATH.read_text())
            self._snapshots = OrderedDict(data.get("snapshots", {}))
            self._deltas = OrderedDict(data.get("deltas", {}))
            self._stats = data.get("stats", self._stats)
            while len(self._snapshots) > self._max_snapshots:
                old_key, _ = self._snapshots.popitem(last=False)
                self._deltas.pop(old_key, None)
        except Exception as e:
            logger.warning("[DeltaEncoder] Failed to load: %s", e)


# Global instance
_delta_encoder: CodeDeltaEncoder | None = None


def get_delta_encoder() -> CodeDeltaEncoder:
    """Get or create the global delta encoder."""
    global _delta_encoder
    if _delta_encoder is None:
        _delta_encoder = CodeDeltaEncoder()
        _delta_encoder.load_from_disk()
    return _delta_encoder
