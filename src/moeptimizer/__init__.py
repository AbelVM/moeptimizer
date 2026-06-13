"""MoE Optimizer — Agentic context optimization middleware.

Transparent OpenAI API proxy that optimizes context for Qwen3.6-35B-A3B-MTP
and similar MoE + MTP models. Preserves reasoning continuity while compressing
stale context to prevent KV-cache thrashing.
"""

from __future__ import annotations

__version__ = "0.1.0"
