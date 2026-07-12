"""TokenCounter — Estimate token usage for context budget management.

Counts tokens with the model's real tokenizer when one is available.

Tokenizer selection (review §6 bug #1): the backend model is Qwen, whose BPE
differs from GPT-4's ``cl100k_base``. When a Qwen tokenizer (HF repo id or local
``tokenizer.json``) is configured, it is loaded via ``transformers`` and used for
exact counts. Otherwise we fall back to tiktoken ``cl100k_base``, which only
*approximates* Qwen counts. In every case the optimizer's runtime calibration
(learned from the backend's real ``prompt_tokens``) corrects the residual ratio.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Callable, ClassVar

import tiktoken

logger = logging.getLogger(__name__)

# Known tiktoken encoding names; anything else is treated as an HF tokenizer id/path.
_TIKTOKEN_ENCODINGS = {"cl100k_base", "o200k_base", "p50k_base", "r50k_base", "gpt2"}


@lru_cache(maxsize=8)
def _load_hf_encode(name_or_path: str, local_only: bool) -> Callable[[str], list[int]] | None:
    """Load and cache an HF tokenizer encode function (or None if unavailable).

    Cached module-wide so repeatedly constructing ``TokenCounter`` instances
    (tests, multiple sessions) does not reload the tokenizer each time.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(
            name_or_path, local_files_only=local_only, trust_remote_code=False
        )
    except Exception:
        return None

    def _encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    return _encode


class TokenCounter:
    """
    Token counter for context budget management.

    Uses the model's real tokenizer (Qwen via ``transformers``) when configured,
    else tiktoken ``cl100k_base`` as an approximation. See module docstring.
    """

    CHARS_PER_TOKEN: ClassVar[dict[str, float]] = {
        "python": 4.0,
        "javascript": 3.8,
        "typescript": 3.7,
        "go": 3.9,
        "rust": 3.6,
        "cpp": 3.5,
        "java": 3.6,
        "c_sharp": 3.5,
        "php": 3.8,
        "ruby": 4.1,
        "html": 2.5,
        "css": 2.8,
        "json": 2.0,
        "generic": 3.5,
    }

    def __init__(self, tokenizer: str = "auto", model_name: str = "gpt-4", max_cache: int = 256) -> None:
        """Initialize the counter.

        Args:
            tokenizer: ``"auto"`` (try a local Qwen HF tokenizer, else tiktoken),
                a HuggingFace repo id / local path (loaded via ``transformers``),
                or a tiktoken encoding name (e.g. ``"cl100k_base"``).
            model_name: legacy hint used only when ``tokenizer == "cl100k_base"``
                style tiktoken selection is requested via a model name.
        """
        self._model_name = model_name
        self._tokenizer_spec = tokenizer
        self._encode: Callable[[str], list[int]] | None = None
        self._backend_name = "unknown"
        # Per-instance fingerprint cache for count_messages (review §7): the
        # stable prefix is re-counted every turn and per-pair inside eviction,
        # so memoizing by content fingerprint avoids re-tokenizing it. Bounded
        # LRU; the fingerprint is a cheap content hash, not a tokenization.
        self._max_cache = max(1, max_cache)
        self._cache: dict[str, int] = {}
        self._cache_order: list[str] = []

        # 1) Explicit HF tokenizer id/path (or "auto" -> try common Qwen ids).
        #    "auto" is restricted to LOCAL files only so it never triggers a
        #    surprise network download on a local/offline box; an explicitly
        #    configured id/path may download.
        candidates: list[tuple[str, bool]] = []  # (name_or_path, local_only)
        if tokenizer and tokenizer not in _TIKTOKEN_ENCODINGS and tokenizer != "auto":
            candidates.append((tokenizer, False))
        if tokenizer == "auto":
            candidates.extend(
                [
                    ("Qwen/Qwen2.5-Coder-32B-Instruct", True),
                    ("Qwen/Qwen2.5-32B", True),
                ]
            )

        for cand, local_only in candidates:
            hf = self._try_load_hf(cand, local_only=local_only)
            if hf is not None:
                self._encode = hf
                self._backend_name = f"hf:{cand}"
                logger.info("TokenCounter using HF tokenizer: %s", cand)
                return

        # 2) tiktoken fallback (explicit encoding name, model name, or default).
        enc = None
        if tokenizer in _TIKTOKEN_ENCODINGS:
            enc = self._try_tiktoken_encoding(tokenizer)
        if enc is None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = self._try_tiktoken_encoding("cl100k_base")
        if enc is None:
            raise RuntimeError(
                "Failed to initialize any tokenizer. Ensure tiktoken is installed."
            )
        self._encode = enc.encode
        self._backend_name = "tiktoken:cl100k_base"
        if tokenizer == "auto":
            logger.warning(
                "TokenCounter falling back to tiktoken cl100k_base (GPT-4 BPE); "
                "this only approximates Qwen token counts. Set server.tokenizer "
                "to a local Qwen tokenizer path for exact counts. Runtime "
                "calibration will correct the ratio from backend prompt_tokens."
            )

    @staticmethod
    def _try_load_hf(name_or_path: str, local_only: bool = False) -> Callable[[str], list[int]] | None:
        """Return a cached encode function for an HF tokenizer, or None.

        ``local_only=True`` restricts loading to already-present local files so
        this never triggers a network download (used for the "auto" default).
        """
        return _load_hf_encode(name_or_path, local_only)

    @staticmethod
    def _try_tiktoken_encoding(name: str) -> "tiktoken.Encoding | None":
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            return None

    @property
    def backend_name(self) -> str:
        """Human-readable name of the active tokenizer backend."""
        return self._backend_name

    def count(self, text: str, lang: str = "generic") -> int:
        """Estimate token count for the given text."""
        if not text:
            return 0

        non_ws = len(text.strip())
        if non_ws == 0:
            return 0

        # Use actual tokenizer
        try:
            return len(self._encode(text))  # type: ignore[misc]
        except Exception:
            # Fallback to character-based estimation
            cpt = self.CHARS_PER_TOKEN.get(lang, 3.5)
            return max(1, int(len(text) / cpt))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens across all messages.

        Memoized by a cheap content fingerprint (review §7): the stable prefix is
        re-counted every turn and per-pair inside eviction, so caching the count
        avoids re-tokenizing identical content repeatedly. The fingerprint is a
        hash of (role, content), not a tokenization, so a cache miss is still only
        a hash. Bounded LRU keeps memory flat across long sessions.
        """
        if not messages:
            return 0
        fp = self._fingerprint(messages)
        cached = self._cache.get(fp)
        if cached is not None:
            # Move to end (most-recently-used) without full re-insertion cost.
            if self._cache_order and self._cache_order[-1] != fp:
                try:
                    self._cache_order.remove(fp)
                except ValueError:
                    pass
                self._cache_order.append(fp)
            return cached

        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += self.count(part.get("text", ""))
            total += 5  # Per-message overhead
        self._cache[fp] = total
        self._cache_order.append(fp)
        if len(self._cache_order) > self._max_cache:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return total

    @staticmethod
    def _fingerprint(messages: list[dict[str, Any]]) -> str:
        """Cheap content fingerprint for the message list (role + content only)."""
        h = hashlib.sha1()
        for msg in messages:
            h.update(msg.get("role", "").encode("utf8"))
            h.update(b"\x1f")
            content = msg.get("content", "")
            if isinstance(content, str):
                h.update(content.encode("utf8"))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        h.update(part.get("text", "").encode("utf8"))
            h.update(b"\x1e")
        return h.hexdigest()

    def count_tokens_precise(self, text: str) -> int:
        """Get precise token count using the model tokenizer."""
        try:
            return len(self._encode(text))  # type: ignore[misc]
        except Exception:
            # Fallback
            return self.count(text)

    def estimate_kv_cache_usage(self, token_count: int) -> str:
        """Convert token count to a human-readable KV-cache estimate."""
        slots = token_count * 4
        if token_count < 10000:
            return f"{token_count:,} tokens (~{slots:,} KV slots)"
        return f"{token_count:,} tokens (~{slots:,} KV slots — near context limit)"