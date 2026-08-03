"""FastAPI application — OpenAI-compatible proxy with agentic context optimization."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import signal
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from openai import APIError, APIStatusError, AsyncOpenAI

from moeptimizer import __version__
from moeptimizer.backend_client import LemonadeClient
from moeptimizer.config import AppConfig, get_config
from moeptimizer.content_store import EXPAND_TOOL_NAME, expand_content_tool
from moeptimizer.embedding import EmbeddingService
from moeptimizer.optimizer import AgentContextOptimizer
from moeptimizer.output_shaper import OutputShaper
from moeptimizer.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Dedicated bounded executor for the CPU-bound optimizer (review §9). Running the
# optimizer on the default ThreadPoolExecutor shares threads with async-IO / embedding
# workers, so under concurrent agentic sessions the optimizer queues and raises TTFT.
# A separate executor isolates optimizer work and bounds its concurrency.
_OPTIMIZER_EXECUTOR: ThreadPoolExecutor | None = None


class _ProxyMetrics:
    """Process-wide aggregate metrics for the proxy (review §11.1).

    Fed from the backend's real ``cached_tokens`` signal on every turn so
    operators can see whether the proxy is actually helping (prefix-cache reuse,
    token savings, latency delta). Cheap, lock-protected counters; no per-turn
    allocation on the request path beyond a couple of integer adds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._last_activity_at: float | None = None
        self.requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cached_tokens = 0
        self.total_prompt_tokens = 0
        self.total_saved_tokens = 0
        self.total_latency_ms = 0.0
        # Real time-to-first-token (review §4.12.1): wall time from request start to
        # the first streamed CONTENT chunk. Distinct from total_latency_ms (the full
        # end-to-end turn time, which the dashboard previously mislabeled as TTFT).
        # ttft_samples counts turns that actually measured a TTFT (streaming only),
        # so avg_ttft_ms averages over those, not over non-streaming turns.
        self.total_ttft_ms = 0.0
        self.ttft_samples = 0
        self.total_completion_tokens = 0
        self.total_completion_duration_ms = 0.0
        self.completion_tps_samples = 0
        # Fresh prefill (review §4.11.2 / Forward plan D1): prompt_tokens -
        # cached_tokens, the tokens the backend re-prefilled this turn. Averaged
        # alongside TTFT so operators see the cache -> TTFT link directly (a rising
        # avg_fresh_prefill_tokens means the prefix is being invalidated more).
        self.total_fresh_prefill_tokens = 0
        self.fresh_prefill_samples = 0
        # Count of turns where the backend returned an error (e.g. HTTP 500 for a
        # truncated tool call) while streaming/serving. Surfaced in /v1/metrics so
        # operators can distinguish "proxy not helping" from "backend failing".
        self.backend_errors = 0
        self.degradation_counts: dict[str, int] = {}
        self.mtp_samples = 0
        self.total_mtp_accepted_tokens = 0
        self.total_mtp_draft_tokens = 0
        self.total_mtp_fallbacks = 0
        self.total_mtp_decode_ms = 0.0
        self.mtp_decode_samples = 0
        self.total_optimizer_ms = 0.0
        self.optimizer_samples = 0
        self.total_token_count_ms = 0.0
        self.token_count_samples = 0
        self._request_traces: list[dict[str, Any]] = []
        self._max_request_traces = 512
        # Per-session breakdown, bounded LRU so it can never grow without limit
        # even under a flood of distinct session ids (review §11.1).
        self._per_session: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_sessions_tracked = 512

    def record_backend_error(self, session_id: str | None = None) -> None:
        """Record that the backend failed to serve a turn (best-effort counter)."""
        with self._lock:
            self.backend_errors += 1
            if session_id:
                entry = self._per_session.get(session_id)
                if entry is None:
                    entry = {
                        "requests": 0,
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "total_cached_tokens": 0,
                        "total_prompt_tokens": 0,
                        "total_saved_tokens": 0,
                        "total_latency_ms": 0.0,
                        "total_ttft_ms": 0.0,
                        "ttft_samples": 0,
                        "backend_errors": 0,
                    }
                    self._per_session[session_id] = entry
                entry["backend_errors"] = entry.get("backend_errors", 0) + 1
                self._per_session.move_to_end(session_id)
                while len(self._per_session) > self._max_sessions_tracked:
                    self._per_session.popitem(last=False)

    def record_degradations(self, counts: dict[str, int]) -> None:
        """Aggregate optimizer stage failures for the process-wide metrics view."""
        if not counts:
            return
        with self._lock:
            for stage, count in counts.items():
                if count > 0:
                    self.degradation_counts[stage] = self.degradation_counts.get(stage, 0) + count

    def record_mtp_usage(
        self,
        usage: Any,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        decode_ms: float | None = None,
    ) -> None:
        """Record optional backend-reported speculative decoding counters."""
        values = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
        details = values.get("completion_tokens_details", {}) or {}
        if not isinstance(details, dict):
            details = getattr(details, "__dict__", {})
        accepted = values.get("accepted_prediction_tokens", details.get("accepted_prediction_tokens"))
        drafted = values.get("draft_tokens", details.get("draft_tokens"))
        fallback = values.get("fallback_count", details.get("fallback_count"))
        reported_decode_ms = values.get("decode_ms", details.get("decode_ms"))
        if not isinstance(accepted, int) and not isinstance(drafted, int) and not isinstance(fallback, int):
            return
        with self._lock:
            self.mtp_samples += 1
            self.total_mtp_accepted_tokens += max(0, accepted) if isinstance(accepted, int) else 0
            self.total_mtp_draft_tokens += max(0, drafted) if isinstance(drafted, int) else 0
            self.total_mtp_fallbacks += max(0, fallback) if isinstance(fallback, int) else 0
            decode_value = decode_ms if decode_ms is not None else reported_decode_ms
            if isinstance(decode_value, (int, float)) and decode_value >= 0:
                self.total_mtp_decode_ms += float(decode_value)
                self.mtp_decode_samples += 1
            if request_id:
                self._append_trace_locked({
                    "request_id": request_id,
                    "session_id": session_id,
                    "mtp": {
                        "draft_tokens": max(0, drafted) if isinstance(drafted, int) else None,
                        "accepted_tokens": max(0, accepted) if isinstance(accepted, int) else None,
                        "fallback_count": max(0, fallback) if isinstance(fallback, int) else None,
                        "decode_ms": max(0.0, decode_ms) if isinstance(decode_ms, (int, float)) else None,
                    },
                })

    def _append_trace_locked(self, trace: dict[str, Any]) -> None:
        self._request_traces.append(trace)
        if len(self._request_traces) > self._max_request_traces:
            del self._request_traces[: len(self._request_traces) - self._max_request_traces]

    def record_optimizer_duration(self, duration_ms: float) -> None:
        with self._lock:
            self.total_optimizer_ms += max(0.0, duration_ms)
            self.optimizer_samples += 1

    def record_token_count_duration(self, duration_ms: float, samples: int) -> None:
        with self._lock:
            self.total_token_count_ms += max(0.0, duration_ms)
            self.token_count_samples += max(0, samples)

    def record_turn(
        self,
        *,
        session_id: str | None = None,
        cached_tokens: int | None = None,
        prompt_tokens: int | None = None,
        saved_tokens: int | None = None,
        latency_ms: float | None = None,
        ttft_ms: float | None = None,
        completion_tokens: int | None = None,
        completion_duration_ms: float | None = None,
        request_id: str | None = None,
        prompt_hash: str | None = None,
        slot: int | None = None,
    ) -> None:
        with self._lock:
            self._last_activity_at = time.time()
            self.requests += 1
            if cached_tokens is not None:
                if cached_tokens > 0:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                self.total_cached_tokens += max(0, cached_tokens)
            if prompt_tokens is not None:
                self.total_prompt_tokens += max(0, prompt_tokens)
            if saved_tokens is not None:
                self.total_saved_tokens += max(0, saved_tokens)
            if latency_ms is not None:
                self.total_latency_ms += max(0.0, latency_ms)
            if ttft_ms is not None:
                self.total_ttft_ms += max(0.0, ttft_ms)
                self.ttft_samples += 1
            if completion_tokens is not None and completion_duration_ms is not None and completion_duration_ms > 0:
                self.total_completion_tokens += max(0, completion_tokens)
                self.total_completion_duration_ms += completion_duration_ms
                self.completion_tps_samples += 1
            if prompt_tokens is not None and cached_tokens is not None:
                self.total_fresh_prefill_tokens += max(0, prompt_tokens - cached_tokens)
                self.fresh_prefill_samples += 1
            if session_id:
                self._record_session_locked(
                    session_id,
                    cached_tokens=cached_tokens,
                    prompt_tokens=prompt_tokens,
                    saved_tokens=saved_tokens,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                )
            if request_id:
                self._append_trace_locked({
                    "request_id": request_id,
                    "session_id": session_id,
                    "prompt_hash": prompt_hash,
                    "slot": slot,
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "fresh_prefill_tokens": (
                        max(0, prompt_tokens - cached_tokens)
                        if prompt_tokens is not None and cached_tokens is not None
                        else None
                    ),
                    "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
                    "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                })

    def request_traces(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(trace) for trace in self._request_traces]

    def _record_session_locked(
        self,
        session_id: str,
        *,
        cached_tokens: int | None,
        prompt_tokens: int | None,
        saved_tokens: int | None,
        latency_ms: float | None,
        ttft_ms: float | None = None,
    ) -> None:
        """Update the per-session counters. Caller must hold ``self._lock``."""
        entry = self._per_session.get(session_id)
        if entry is None:
            entry = {
                "requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_cached_tokens": 0,
                "total_prompt_tokens": 0,
                "total_saved_tokens": 0,
                "total_latency_ms": 0.0,
                "total_ttft_ms": 0.0,
                "ttft_samples": 0,
                "total_fresh_prefill_tokens": 0,
                "fresh_prefill_samples": 0,
                "backend_errors": 0,
            }
        entry["requests"] += 1
        if cached_tokens is not None:
            if cached_tokens > 0:
                entry["cache_hits"] += 1
            else:
                entry["cache_misses"] += 1
            entry["total_cached_tokens"] += max(0, cached_tokens)
        if prompt_tokens is not None:
            entry["total_prompt_tokens"] += max(0, prompt_tokens)
        if saved_tokens is not None:
            entry["total_saved_tokens"] += max(0, saved_tokens)
        if latency_ms is not None:
            entry["total_latency_ms"] += max(0.0, latency_ms)
        if ttft_ms is not None:
            entry["total_ttft_ms"] = entry.get("total_ttft_ms", 0.0) + max(0.0, ttft_ms)
            entry["ttft_samples"] = entry.get("ttft_samples", 0) + 1
        if prompt_tokens is not None and cached_tokens is not None:
            entry["total_fresh_prefill_tokens"] = entry.get("total_fresh_prefill_tokens", 0) + max(
                0, prompt_tokens - cached_tokens
            )
            entry["fresh_prefill_samples"] = entry.get("fresh_prefill_samples", 0) + 1
        # Move-to-end keeps most-recently-active sessions; evict the oldest.
        self._per_session[session_id] = entry
        self._per_session.move_to_end(session_id)
        while len(self._per_session) > self._max_sessions_tracked:
            self._per_session.popitem(last=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = self.requests
            hits = self.cache_hits
            cached = self.total_cached_tokens
            prefill_tokens = cached + self.total_fresh_prefill_tokens
            reuse_ratio = (cached / prefill_tokens) if prefill_tokens else 0.0
            uptime_seconds = max(0.0, time.time() - self._started_at)
            saved_basis_tokens = self.total_prompt_tokens + self.total_saved_tokens
            sessions: dict[str, Any] = {}
            for sid, e in self._per_session.items():
                s_req = e["requests"]
                s_cached = e["total_cached_tokens"]
                s_prompt = e["total_prompt_tokens"]
                s_ttft_samples = e.get("ttft_samples", 0)
                sessions[sid] = {
                    "requests": s_req,
                    "cache_hits": e["cache_hits"],
                    "cache_misses": e["cache_misses"],
                    "cache_hit_rate": round(e["cache_hits"] / max(1, s_req), 4),
                    "total_cached_tokens": s_cached,
                    "total_prompt_tokens": s_prompt,
                    "prefix_cache_reuse_ratio": round(
                        (
                            s_cached
                            / max(1, s_cached + e.get("total_fresh_prefill_tokens", 0))
                        )
                        if s_cached or e.get("total_fresh_prefill_tokens", 0)
                        else 0.0,
                        4,
                    ),
                    "total_saved_tokens": e["total_saved_tokens"],
                    "avg_latency_ms": round(e["total_latency_ms"] / max(1, s_req), 1),
                    "avg_ttft_ms": round(
                        e.get("total_ttft_ms", 0.0) / max(1, s_ttft_samples), 1
                    ),
                    "avg_fresh_prefill_tokens": round(
                        e.get("total_fresh_prefill_tokens", 0)
                        / max(1, e.get("fresh_prefill_samples", 0)),
                        1,
                    ),
                    "backend_errors": e.get("backend_errors", 0),
                }
            return {
                "metrics_window_started_at": self._started_at,
                "last_activity_at": self._last_activity_at,
                "uptime_seconds": round(uptime_seconds, 1),
                "requests_per_minute": round(requests / max(1.0, uptime_seconds / 60.0), 2),
                "last_activity_age_seconds": (
                    round(max(0.0, time.time() - self._last_activity_at), 1)
                    if self._last_activity_at is not None
                    else None
                ),
                "requests": requests,
                "cache_hits": hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(hits / max(1, requests), 4),
                "total_cached_tokens": cached,
                "total_prompt_tokens": self.total_prompt_tokens,
                "prefix_cache_reuse_ratio": round(reuse_ratio, 4),
                "total_saved_tokens": self.total_saved_tokens,
                "total_raw_input_tokens": saved_basis_tokens,
                "input_savings_ratio": round(
                    self.total_saved_tokens / saved_basis_tokens, 4
                ) if saved_basis_tokens else 0.0,
                "total_latency_ms": round(self.total_latency_ms, 1),
                "backend_error_rate": round(self.backend_errors / max(1, requests), 4),
                "avg_latency_ms": round(self.total_latency_ms / max(1, requests), 1),
                # Real time-to-first-token averaged over streaming turns that
                # measured it (review §4.12.1); avg_latency_ms is the full turn time.
                "avg_ttft_ms": round(
                    self.total_ttft_ms / max(1, self.ttft_samples), 1
                ),
                "ttft_samples": self.ttft_samples,
                "avg_completion_tps": round(
                    self.total_completion_tokens
                    / max(0.001, self.total_completion_duration_ms / 1000.0),
                    2,
                ) if self.completion_tps_samples else None,
                "completion_tps_samples": self.completion_tps_samples,
                # Fresh prefill (prompt - cached) averaged over turns that reported
                # both; read next to avg_ttft_ms to see the cache -> TTFT link.
                "avg_fresh_prefill_tokens": round(
                    self.total_fresh_prefill_tokens / max(1, self.fresh_prefill_samples), 1
                ),
                "fresh_prefill_samples": self.fresh_prefill_samples,
                "backend_errors": self.backend_errors,
                "degradation_counts": dict(self.degradation_counts),
                "mtp_samples": self.mtp_samples,
                "avg_mtp_accepted_tokens": round(
                    self.total_mtp_accepted_tokens / max(1, self.mtp_samples), 1
                ),
                "avg_mtp_draft_tokens": round(
                    self.total_mtp_draft_tokens / max(1, self.mtp_samples), 1
                ),
                "total_mtp_accepted_tokens": self.total_mtp_accepted_tokens,
                "total_mtp_draft_tokens": self.total_mtp_draft_tokens,
                "total_mtp_fallbacks": self.total_mtp_fallbacks,
                "mtp_fallback_rate": round(
                    self.total_mtp_fallbacks / max(1, self.mtp_samples), 4
                ),
                "avg_mtp_decode_ms": round(
                    self.total_mtp_decode_ms / max(1, self.mtp_decode_samples), 1
                ),
                "mtp_decode_samples": self.mtp_decode_samples,
                "mtp_acceptance_rate": round(
                    self.total_mtp_accepted_tokens / max(1, self.total_mtp_draft_tokens), 4
                ),
                "avg_optimizer_ms": round(
                    self.total_optimizer_ms / max(1, self.optimizer_samples), 1
                ),
                "optimizer_samples": self.optimizer_samples,
                "avg_token_count_ms": round(
                    self.total_token_count_ms / max(1, self.token_count_samples), 3
                ),
                "token_count_samples": self.token_count_samples,
                "sessions": sessions,
            }

    def reset(self) -> None:
        with self._lock:
            self.__init__()


# Single process-wide metrics instance.
PROXY_METRICS = _ProxyMetrics()

# Self-contained live dashboard HTML (review §11 / P4c). No framework, no external
# assets. Browser-side history powers the trend chart because the live API exposes
# aggregate snapshots, not benchmark-style per-turn time series.
_METRICS_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOEptimizer - Live Operations</title>
<style>
    :root { --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --ink:#c9d1d9;
                    --muted:#8b949e; --line:#30363d; --mint:#3fb950; --blue:#58a6ff;
                    --amber:#e3b341; --red:#f85149; --shadow:0 12px 30px rgba(0,0,0,.2); }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 85% 0%,#182b35 0,#0d1117 38%);
                 color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
    header { max-width:1440px; margin:auto; padding:22px clamp(18px,4vw,52px) 18px;
                     display:flex; justify-content:space-between; align-items:flex-end; gap:18px; }
    .brand { display:flex; align-items:center; gap:12px; }
    .mark { width:12px; height:42px; background:var(--blue); transform:skew(-14deg); }
    h1,h2,p { margin:0; }
    h1 { font-size:18px; line-height:1.05; font-weight:600; letter-spacing:.2px; }
    .sub { color:var(--muted); margin-top:5px; font-size:12px; }
    .live { display:flex; align-items:center; gap:8px; color:var(--mint); font-size:12px; }
    .dot { width:8px; height:8px; border-radius:50%; background:currentColor; }
    .live.stale { color:var(--amber); }
    .live.down { color:var(--red); }
    button { border:1px solid var(--line); background:transparent; color:var(--muted); padding:7px 10px;
                     font:12px inherit; cursor:pointer; }
    button:hover { color:var(--ink); border-color:var(--mint); }
    main { max-width:1440px; margin:auto; padding:10px clamp(18px,4vw,52px) 36px; }
    .hero { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(280px,.65fr); gap:18px;
                    padding:26px 0 30px; }
    .hero h2 { font-size:clamp(28px,5vw,48px); line-height:1.02; font-weight:600; max-width:720px; }
    .hero p { color:var(--muted); max-width:560px; margin-top:14px; }
    .hero-note { border-left:2px solid var(--blue); padding:8px 0 8px 18px; align-self:end; }
    .hero-note strong { display:block; color:var(--blue); font-size:26px; font-weight:500; }
    .hero-note span { color:var(--muted); font-size:12px; }
    .hero-meta { color:var(--muted); font-size:11px; margin-top:5px; }
    .section { color:var(--blue); font-size:11px; font-weight:700; letter-spacing:.12em;
                         text-transform:uppercase; border-bottom:1px solid var(--line); padding-bottom:8px; margin:4px 0 12px; }
    .cards { display:grid; grid-template-columns:repeat(7,minmax(110px,1fr)); gap:10px; }
    .stat { background:var(--panel); border:1px solid var(--line); padding:15px 14px; min-height:108px; box-shadow:var(--shadow); }
    .stat .v { font-size:clamp(22px,3vw,30px); line-height:1.05; font-weight:600; margin-bottom:9px; }
    .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
    .stat-detail { color:var(--muted); font-size:11px; margin-top:5px; }
    .stat.mint .v { color:var(--mint); } .stat.blue .v { color:var(--blue); }
    .stat.amber .v { color:var(--amber); } .stat.red .v { color:var(--red); }
    .layout { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(300px,.6fr); gap:18px; margin-top:18px; }
    .panel { background:var(--panel); border:1px solid var(--line); padding:18px; min-width:0; }
    .panel h2 { font-size:16px; line-height:1.2; font-weight:600; }
    .hint { color:var(--muted); font-size:12px; margin:5px 0 14px; }
    #trend,.evolution-chart { width:100%; height:250px; display:block; overflow:visible; }
    .legend { display:flex; gap:15px; color:var(--muted); font-size:11px; margin-top:5px; }
    .legend i { display:inline-block; width:9px; height:9px; margin-right:5px; background:var(--mint); }
    .legend i.blue { background:var(--blue); }
    .ops { display:grid; gap:12px; }
    .op { border-top:1px solid var(--line); padding-top:10px; }
    .op:first-child { border-top:0; padding-top:0; }
    .op-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
    .op-value { font-size:18px; margin-top:2px; }
    .op-detail { color:var(--muted); font-size:12px; }
    .wide { grid-column:1/-1; }
    .meta-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:0 0 18px; }
    .meta-item { background:var(--panel); border:1px solid var(--line); padding:11px 13px; min-width:0; }
    .meta-item .k { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.06em; }
    .meta-item .v { margin-top:4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .chart-stack { display:grid; gap:10px; }
    .chart-stack h3 { font-size:12px; font-weight:600; margin:4px 0 -4px; }
    .evolution { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; margin-top:18px; }
    .chart-note { color:var(--muted); font-size:11px; margin-top:4px; }
    .tablewrap { overflow:auto; max-height:390px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th,td { text-align:right; padding:8px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
    th { color:var(--muted); font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
    th:first-child,td:first-child { text-align:left; }
    td:first-child { color:var(--ink); font-family:ui-monospace,SFMono-Regular,monospace; }
    tr:hover { background:var(--panel2); }
    .barwrap { display:inline-block; width:90px; height:6px; background:var(--line); vertical-align:middle; }
    .bar { height:100%; background:var(--mint); }
    .empty { color:var(--muted); font-style:italic; padding:18px 0; }
    .degradation { color:var(--amber); font-family:ui-monospace,SFMono-Regular,monospace; font-size:12px; }
    footer { max-width:1440px; margin:auto; padding:12px clamp(18px,4vw,52px) 24px; color:var(--muted); font-size:11px; }
    code { color:var(--blue); }
    @media (max-width:1100px) { .evolution { grid-template-columns:1fr; } }
    @media (max-width:900px) { .cards { grid-template-columns:repeat(3,1fr); } .hero,.layout { grid-template-columns:1fr; } .meta-grid { grid-template-columns:repeat(2,1fr); } }
    @media (max-width:540px) { .cards { grid-template-columns:repeat(2,1fr); } header { align-items:flex-start; flex-direction:column; } }
    @media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; } }
</style>
</head>
<body>
<header>
    <div class="brand"><span class="mark" aria-hidden="true"></span><div><h1>MOEptimizer</h1><div class="sub">Live inference operations</div></div></div>
    <div class="live"><div id="status" class="live"><span class="dot" aria-hidden="true"></span><span>Connecting</span></div><button id="reset" type="button">Reset metrics</button></div>
</header>
<main>
    <section class="hero"><div><h2 id="headline">Watching the prefix hold.</h2><p id="summary">Waiting for the first metrics snapshot.</p></div><div class="hero-note"><strong id="s-requests">0</strong><span>completed requests in this process</span><div class="hero-meta"><span id="s-rate">—</span> requests/min</div></div></section>
    <div class="section">Core signal</div>
    <section class="cards">
        <div class="stat mint"><div class="v" id="s-reuse">—</div><div class="k">Prefix reuse</div></div>
        <div class="stat blue"><div class="v" id="s-saved">—</div><div class="k">Tokens saved</div><div class="stat-detail" id="s-saved-rate">— of raw input</div></div>
        <div class="stat"><div class="v" id="s-hitrate">—</div><div class="k">Cache hit rate</div></div>
        <div class="stat amber"><div class="v" id="s-ttft">—</div><div class="k">Average TTFT</div></div>
        <div class="stat mint"><div class="v" id="s-tps">—</div><div class="k">Decode TPS</div></div>
        <div class="stat"><div class="v" id="s-prefill">—</div><div class="k">Fresh prefill</div></div>
        <div class="stat red"><div class="v" id="s-err">—</div><div class="k">Backend errors</div></div>
    </section>
    <div class="section">Runtime identity</div>
    <section class="meta-grid">
        <div class="meta-item"><div class="k">Proxy version</div><div class="v" id="meta-version">—</div></div>
        <div class="meta-item"><div class="k">LLM model</div><div class="v" id="meta-llm">—</div></div>
        <div class="meta-item"><div class="k">Embedding model</div><div class="v" id="meta-embed">—</div></div>
        <div class="meta-item"><div class="k">Metrics cadence</div><div class="v">3 seconds</div></div>
    </section>
    <section class="layout">
        <div class="panel"><h2>Cache pressure over time</h2><p class="hint">Browser history of the last 30 snapshots. The three charts share the same snapshot timeline.</p><div class="chart-stack"><h3>Cache pressure</h3><svg id="trend" viewBox="0 0 520 250" role="img" aria-label="Cache reuse and fresh prefill trend"></svg><div class="legend"><span><i></i>prefix reuse</span><span><i class="blue"></i>fresh prefill, normalized</span></div><h3>Tokens saved - absolute</h3><svg id="saved-trend" class="evolution-chart" viewBox="0 0 520 250" role="img" aria-label="Cumulative tokens saved in absolute tokens"></svg><div class="chart-note" id="saved-note">Higher is better</div><h3>Tokens saved - ratio</h3><svg id="saved-ratio-trend" class="evolution-chart" viewBox="0 0 520 250" role="img" aria-label="Token savings percentage of raw input trend"></svg><div class="chart-note">Saved as a percentage of raw input</div></div></div>
        <div class="panel"><h2>Runtime pulse</h2><p class="hint">Only signals emitted by the live proxy are shown.</p><div class="ops"><div class="op"><div class="op-label">Average latency</div><div class="op-value" id="s-latency">—</div><div class="op-detail">full request duration</div></div><div class="op"><div class="op-label">Input savings</div><div class="op-value" id="s-savings">—</div><div class="op-detail">saved tokens / original input estimate</div></div><div class="op"><div class="op-label">Backend error rate</div><div class="op-value" id="s-error-rate">—</div><div class="op-detail">failed turns / completed requests</div></div><div class="op"><div class="op-label">Metrics window</div><div class="op-value" id="s-window-age">—</div><div class="op-detail">time since reset or process start</div></div><div class="op"><div class="op-label">Last backend turn</div><div class="op-value" id="s-last-activity">—</div><div class="op-detail">actual successful proxy activity</div></div><div class="op"><div class="op-label">Optimizer overhead</div><div class="op-value" id="s-optimizer">—</div><div class="op-detail">average proxy-side optimization</div></div><div class="op"><div class="op-label">Token counting</div><div class="op-value" id="s-token-count">—</div><div class="op-detail">average counting cost</div></div><div class="op"><div class="op-label">MTP</div><div class="op-value" id="s-mtp">—</div><div class="op-detail" id="s-mtp-detail">No MTP usage reported yet.</div></div><div class="op"><div class="op-label">Degraded stages</div><div class="op-value degradation" id="s-degraded">None</div></div></div></div>
        <div class="panel wide"><h2>Active sessions</h2><p class="hint">Bounded process-local view, sorted by most recently active. Session IDs are intentionally shortened.</p><div id="sess"><p class="empty">No sessions recorded yet.</p></div></div>
    </section>
    <div class="section" style="margin-top:24px">Performance</div>
    <section class="evolution">
        <div class="panel"><h2>TTFT</h2><p class="hint">Average time to first content token across completed streams.</p><svg id="ttft-trend" class="evolution-chart" viewBox="0 0 520 250" role="img" aria-label="Average time to first token trend"></svg><div class="chart-note">Lower is better</div></div>
        <div class="panel"><h2>Decode throughput</h2><p class="hint">Completion tokens per second when backend usage reports completion tokens.</p><svg id="tps-trend" class="evolution-chart" viewBox="0 0 520 250" role="img" aria-label="Completion tokens per second trend"></svg><div class="chart-note" id="tps-note">Waiting for completion-token usage.</div></div>
    </section>
</main>
<footer>Auto-refresh every 3s &middot; updated <span id="updated">never</span> &middot; source: <code>GET /v1/metrics</code></footer>
<script>
const fmt = n => (n==null ? "—" : Number(n).toLocaleString());
const pct = n => (n==null ? "—" : (Number(n)*100).toFixed(1) + "%");
const ms = n => (n==null ? "—" : Number(n).toLocaleString(undefined,{maximumFractionDigits:1}) + " ms");
const age = n => (n==null ? "—" : n < 60 ? Math.round(n) + " s" : n < 3600 ? Math.floor(n/60) + "m " + Math.floor(n%60) + "s" : Math.floor(n/3600) + "h " + Math.floor((n%3600)/60) + "m");
const history = [];
const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function setStatus(kind, text) { const e=document.getElementById("status"); e.className="live " + kind; e.lastElementChild.textContent=text; }
function drawSeries(id, key, color, format, emptyText, axisFontSize=10) {
    const svg=document.getElementById(id), width=520, height=250, pad={l:42,r:12,t:16,b:28};
    const values=history.map(x=>x[key]).filter(v=>Number.isFinite(v));
    svg.innerHTML="";
    if(values.length < 2) { svg.innerHTML='<text x="42" y="125" fill="#8b949e" font-size="12">'+emptyText+'</text>'; return; }
    const min=Math.min(...values), max=Math.max(...values), span=Math.max(1e-9,max-min);
    const innerW=width-pad.l-pad.r, innerH=height-pad.t-pad.b;
    const x=i=>pad.l+(i/(Math.max(1,history.length-1)))*innerW;
    const y=v=>pad.t+(1-(v-min)/span)*innerH;
    const points=history.map((item,i)=>Number.isFinite(item[key]) ? [x(i),y(item[key])] : null).filter(Boolean);
    const path=points.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+","+p[1].toFixed(1)).join(" ");
    const grid=[0,.5,1].map(v=>'<line x1="'+pad.l+'" x2="'+(width-pad.r)+'" y1="'+(pad.t+v*innerH)+'" y2="'+(pad.t+v*innerH)+'" stroke="#30363d"/><text x="4" y="'+(pad.t+v*innerH+4)+'" fill="#8b949e" font-size="'+axisFontSize+'">'+format(max-v*span)+'</text>').join("");
    svg.innerHTML=grid+'<path d="'+path+'" fill="none" stroke="'+color+'" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>';
}
function drawTrend() {
    const svg=document.getElementById("trend"), width=520, height=250, pad={l:42,r:12,t:16,b:28};
    const innerW=width-pad.l-pad.r, innerH=height-pad.t-pad.b;
    svg.innerHTML="";
    if(history.length < 2) { svg.innerHTML='<text x="42" y="125" fill="#96a19d" font-size="12">Collecting snapshots...</text>'; return; }
    const maxFresh=Math.max(1,...history.map(x=>x.fresh));
    const x=i=>pad.l+(i/(history.length-1))*innerW, y=v=>pad.t+(1-v)*innerH;
    const path=key=>history.map((item,i)=>(i?"L":"M")+x(i).toFixed(1)+","+y(key==="reuse"?item.reuse:item.fresh/maxFresh).toFixed(1)).join(" ");
    const grid=[0,.5,1].map(v=>'<line x1="'+pad.l+'" x2="'+(width-pad.r)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#303a3d"/><text x="4" y="'+(y(v)+4)+'" fill="#96a19d" font-size="10">'+Math.round(v*100)+'%</text>').join("");
    svg.innerHTML=grid+'<path d="'+path("reuse")+'" fill="none" stroke="#a7e8c0" stroke-width="2.5"/><path d="'+path("fresh")+'" fill="none" stroke="#8fc7ff" stroke-width="2" stroke-dasharray="5 4"/>';
    drawSeries("saved-trend","saved","#58a6ff",fmt,"Collecting snapshots...");
    drawSeries("saved-ratio-trend","savedPct","#a7e8c0",pct,"Collecting savings percentage...");
    drawSeries("ttft-trend","ttft","#e3b341",ms,"Waiting for streamed TTFT...",12);
    drawSeries("tps-trend","tps","#3fb950",n=>Number(n).toFixed(1),"TPS unavailable",12);
}
function renderSessions(sessions) {
    const ids=Object.keys(sessions||{}), box=document.getElementById("sess");
    if(!ids.length) { box.innerHTML='<p class="empty">No sessions recorded yet.</p>'; return; }
    const maxSaved=Math.max(1,...ids.map(id=>sessions[id].total_saved_tokens||0));
    let rows='<div class="tablewrap"><table><thead><tr><th>session</th><th>requests</th><th>reuse</th><th>fresh prefill</th><th>saved</th><th>latency</th><th>errors</th><th></th></tr></thead><tbody>';
    for(const id of ids.slice().reverse().slice(0,64)) { const s=sessions[id], w=Math.round(100*(s.total_saved_tokens||0)/maxSaved); rows+='<tr><td title="'+esc(id)+'">'+esc(id.slice(0,14))+'</td><td>'+fmt(s.requests)+'</td><td>'+pct(s.prefix_cache_reuse_ratio)+'</td><td>'+fmt(s.avg_fresh_prefill_tokens)+'</td><td>'+fmt(s.total_saved_tokens)+'</td><td>'+ms(s.avg_latency_ms)+'</td><td>'+fmt(s.backend_errors)+'</td><td><span class="barwrap"><span class="bar" style="display:block;width:'+w+'%"></span></span></td></tr>'; }
    box.innerHTML=rows+'</tbody></table></div>';
}
function render(d) {
    const fresh=Number(d.avg_fresh_prefill_tokens||0), reuse=Number(d.prefix_cache_reuse_ratio||0);
    history.push({reuse,fresh,saved:Number(d.total_saved_tokens||0),savedPct:Number(d.input_savings_ratio||0),ttft:Number.isFinite(Number(d.avg_ttft_ms))&&d.ttft_samples?Number(d.avg_ttft_ms):null,tps:d.avg_completion_tps==null?null:Number(d.avg_completion_tps)}); if(history.length>30) history.shift();
    document.getElementById("s-requests").textContent=fmt(d.requests);
    document.getElementById("s-rate").textContent=fmt(d.requests_per_minute);
    document.getElementById("s-reuse").textContent=pct(d.prefix_cache_reuse_ratio);
    document.getElementById("s-saved").textContent=fmt(d.total_saved_tokens);
    document.getElementById("s-saved-rate").textContent=pct(d.input_savings_ratio)+" of raw input";
    document.getElementById("s-hitrate").textContent=pct(d.cache_hit_rate);
    document.getElementById("s-ttft").textContent=ms(d.avg_ttft_ms);
    document.getElementById("s-tps").textContent=d.avg_completion_tps==null?"—":Number(d.avg_completion_tps).toFixed(1);
    document.getElementById("tps-note").textContent=d.completion_tps_samples?"Aggregate completion tokens / measured decode time.":"Waiting for completion-token usage.";
    document.getElementById("s-prefill").textContent=fmt(d.avg_fresh_prefill_tokens);
    document.getElementById("s-err").textContent=fmt(d.backend_errors);
    document.getElementById("s-savings").textContent=pct(d.input_savings_ratio);
    document.getElementById("s-error-rate").textContent=pct(d.backend_error_rate);
    document.getElementById("s-latency").textContent=ms(d.avg_latency_ms);
    document.getElementById("s-window-age").textContent=age(d.uptime_seconds);
    document.getElementById("s-last-activity").textContent=age(d.last_activity_age_seconds);
    document.getElementById("s-optimizer").textContent=ms(d.avg_optimizer_ms);
    document.getElementById("s-token-count").textContent=ms(d.avg_token_count_ms);
    document.getElementById("saved-note").textContent=fmt(d.total_saved_tokens)+" tokens / "+pct(d.input_savings_ratio)+" of raw input";
    document.getElementById("s-mtp").textContent=d.mtp_samples?pct(d.mtp_acceptance_rate):"—";
    document.getElementById("s-mtp-detail").textContent=d.mtp_samples?(fmt(d.avg_mtp_accepted_tokens)+" accepted / "+fmt(d.avg_mtp_draft_tokens)+" drafted; "+pct(d.mtp_fallback_rate)+" fallback") : "No MTP usage reported yet.";
    const degraded=Object.entries(d.degradation_counts||{}); document.getElementById("s-degraded").textContent=degraded.length?degraded.map(x=>x[0]+" ("+x[1]+")").join(", "):"None";
    document.getElementById("meta-version").textContent="v"+"__PROXY_VERSION__";
    document.getElementById("meta-llm").textContent="__LLM_MODEL__";
    document.getElementById("meta-embed").textContent="__EMBED_MODEL__";
    document.getElementById("headline").textContent=d.backend_errors?"The backend needs attention.":(reuse>=.7?"The prefix is holding.":"Watching the prefix hold.");
    document.getElementById("summary").textContent=d.backend_errors?"Backend errors are present; treat latency and cache movement as diagnostic signals.":(fresh>0?"Fresh prefill is the work the model had to repeat on this process.":"No fresh-prefill sample has arrived yet.");
    document.getElementById("updated").textContent=new Date().toLocaleTimeString();
    renderSessions(d.sessions); drawTrend(); setStatus(d.last_activity_age_seconds!=null&&d.last_activity_age_seconds>15?"stale":"",d.last_activity_age_seconds!=null&&d.last_activity_age_seconds>15?"Idle":"Live");
}
async function resetMetrics() {
    const button=document.getElementById("reset"); button.disabled=true;
    try { const r=await fetch("/v1/metrics/reset",{method:"POST"}); if(!r.ok) throw Error(r.status); history.length=0; document.getElementById("summary").textContent="Metrics reset. Waiting for the next snapshot."; drawTrend(); setStatus("","Reset"); }
    catch (e) { setStatus("down","Reset failed"); }
    finally { button.disabled=false; }
}
document.getElementById("reset").addEventListener("click", resetMetrics);
async function refresh() {
    try { const r=await fetch("/v1/metrics",{cache:"no-store"}); if(!r.ok) throw Error(r.status); render(await r.json()); }
    catch (e) { setStatus("down","Metrics unavailable"); document.getElementById("summary").textContent="Could not read /v1/metrics. The dashboard will retry automatically."; }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


def _explain_header_value(messages: list[dict[str, Any]]) -> str:
    """Serialize the optimized prompt for the explain-mode response header.

    Base64-encoded JSON so the value is header-safe regardless of message
    content (newlines, colons, unicode). Decoded on the client with
    ``json.loads(base64.b64decode(value))``.
    """
    import base64

    payload = json.dumps(messages, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _serialize_messages_text(messages: list[dict[str, Any]]) -> str:
    """Render optimized messages as plain text for the faithfulness header.

    Used by the ``X-MOEPT-Optimized-Prompt-Text`` response header so a
    local benchmark can measure how much of the original context survived
    compaction. Newlines are replaced with ``\\n`` to keep the value
    header-safe; the benchmark reverses this on read.
    """
    return _messages_to_text(messages).replace("\n", "\\n")


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Render messages as plain ``[role]\\ncontent`` text joined by newlines.

    The raw (newline-preserving) form used for the proxy-side faithfulness
    computation (#7); ``_serialize_messages_text`` escapes the newlines for the
    HTTP header.
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"[{role}]\n{content}")
    return "\n".join(parts)


def _prompt_faithfulness(full_prompt: str, optimized_prompt: str) -> float | None:
    """Token-set Jaccard between the full and optimized prompt (1.0 = nothing
    lost). Proxy-side mirror of the benchmark metric (#7). None if either empty."""
    if not full_prompt or not optimized_prompt:
        return None
    full_tokens = set(full_prompt.lower().split())
    opt_tokens = set(optimized_prompt.lower().split())
    if not full_tokens or not opt_tokens:
        return None
    return round(len(full_tokens & opt_tokens) / max(len(full_tokens | opt_tokens), 1), 6)


def _prompt_source_token_recall(full_prompt: str, optimized_prompt: str) -> float | None:
    """Fraction of unique source tokens retained after compaction (#7)."""
    if not full_prompt or not optimized_prompt:
        return None
    full_tokens = set(full_prompt.lower().split())
    opt_tokens = set(optimized_prompt.lower().split())
    if not full_tokens or not opt_tokens:
        return None
    return round(len(full_tokens & opt_tokens) / len(full_tokens), 6)


def _evicted_content_recall(full_prompt: str, optimized_prompt: str) -> float | None:
    """Recall of tokens from the evicted (first ~60%) part of the prompt that the
    optimized prompt retained (#7). None when the prompt is too short to split."""
    if not full_prompt or not optimized_prompt:
        return None
    full_tokens = full_prompt.lower().split()
    if len(full_tokens) < 40:
        return None
    split = int(len(full_tokens) * 0.6)
    evicted_tokens = set(full_tokens[:split])
    if not evicted_tokens:
        return None
    opt_tokens = set(optimized_prompt.lower().split())
    return round(len(evicted_tokens & opt_tokens) / max(len(evicted_tokens), 1), 6)


# Common unicode punctuation that appears in code comments / fixture text but
# is not latin-1-encodable. Folded to ASCII so header values stay valid.
_HEADER_UNICODE_FOLD = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # horizontal ellipsis
    "\u00a0": " ",    # non-breaking space
}


def _header_safe(value: str) -> str:
    """Return ``value`` with any non-latin-1 character folded to a safe ASCII
    substitute so it can be placed in an HTTP response header.

    HTTP header values are latin-1-encoded by Starlette; an un-sanitized unicode
    value (em-dash, smart quotes, non-latin scripts) raises UnicodeEncodeError
    and turns the whole response into a 500. We first apply a small punctuation
    fold, then drop any remaining out-of-range code points.
    """
    if not value:
        return value
    folded = "".join(_HEADER_UNICODE_FOLD.get(ch, ch) for ch in value)
    return "".join(ch if ord(ch) <= 255 else "" for ch in folded)


def _dry_run_response(
    optimizer: AgentContextOptimizer,
    original_messages: list[dict[str, Any]],
    optimized_messages: list[dict[str, Any]],
    optimization_error: str | None,
) -> JSONResponse:
    """Build the response for ``X-MOEPT-Dry-Run`` requests (review §11 / P4a).

    Returns the optimized prompt the proxy would send to the backend, a token
    savings diff, and an estimated cache-hit signal — WITHOUT calling the backend.
    Purely observational; never mutates session state or metrics.
    """
    original_tokens = optimizer.token_counter.count_messages(original_messages)
    optimized_tokens = optimizer.token_counter.count_messages(optimized_messages)
    saved = max(0, original_tokens - optimized_tokens)
    # Estimated cache hit: the proxy's static-prefix KV cache key for the optimized
    # prompt. A stable (reused) key across turns is the local proxy signal that the
    # backend prefix cache will be reused. We report the key's presence, not a
    # backend-authorized count (which only the backend can give after a real call).
    cache_key = ""
    with suppress(Exception):
        cache_key = optimizer.get_cache_key(optimized_messages)
    payload: dict[str, Any] = {
        "object": "moept.dry_run",
        "optimized_messages": optimized_messages,
        "tokens": {
            "original": original_tokens,
            "optimized": optimized_tokens,
            "saved": saved,
            "saved_pct": round(100.0 * saved / original_tokens, 1) if original_tokens else 0.0,
        },
        "est_cache_hit": bool(cache_key),
        "cache_key_prefix": cache_key[:16] if cache_key else "",
        "optimization_error": optimization_error,
    }
    return JSONResponse(content=payload)


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    """Validate that all non-assistant messages have a 'content' field.

    The Lemonade server requires all non-assistant messages to contain 'content'.
    This function ensures compliance before sending requests to the backend.

    Raises:
        ValueError: If any non-assistant message is missing 'content'
    """
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role != "assistant" and "content" not in msg:
            raise ValueError(
                f"All non-assistant messages must contain 'content'. "
                f"Message {i} has role='{role}' but no 'content' field."
            )


def _ensure_content(messages: list[dict[str, Any]]) -> None:
    """Ensure all non-assistant messages have a 'content' field (set to '' if missing).

    The Lemonade server requires all non-assistant messages to contain 'content'.
    This is applied after optimization because the optimizer/compactor may produce
    tool_result or other non-assistant messages that lack content.
    """
    for msg in messages:
        role = msg.get("role", "")
        if role != "assistant" and "content" not in msg:
            msg["content"] = ""


def _scrub_internal_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages with proxy-internal ``_*`` keys removed.

    Last-line scrub at the backend boundary (review §4.8.3). The optimizer strips
    internal markers at Step 13, but the volatile trailing turn is appended AFTER
    that (Step 14.12) tagged ``_volatile_turn``, so its marker would otherwise leak
    to the backend — violating the OpenAI-transparent contract and the
    "no model-visible markers" constraint. Returns new dicts so the optimizer's
    stored ``_last_optimized`` (which the incremental path reuses) is untouched.
    """
    return [
        {k: v for k, v in msg.items() if not k.startswith("_")}
        for msg in messages
    ]


def _fallback_optimized_messages(messages: list[dict[str, Any]], keep_full_steps: int) -> list[dict[str, Any]]:
    """Return a safe compact fallback when the full optimizer fails.

    This avoids forwarding the full raw conversation to the backend after an
    optimizer exception. It preserves the system prompt and the most recent
    user/assistant turns, which is safer than sending an unbounded raw context.
    """
    if not messages:
        return []

    fallback: list[dict[str, Any]] = []
    if messages[0].get("role") == "system":
        fallback.append(dict(messages[0]))
        start_index = 1
    else:
        start_index = 0

    keep = max(1, keep_full_steps) * 2
    recent = messages[max(start_index, len(messages) - keep):]
    # Strip internal proxy flags so they never leak to the backend (review §5.6).
    cleaned: list[dict[str, Any]] = []
    for msg in recent:
        cleaned_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
        cleaned.append(cleaned_msg)
    fallback.extend(cleaned)
    return fallback


def _canonicalize(value: Any) -> Any:
    """Return a JSON-stable representation for session fingerprinting."""
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_text(text: str) -> str:
    """Return a compact stable hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _first_user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first user message, used as a stable conversation seed."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg
    return {}


def _resolve_session_id(
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    legacy_session_id: Any = None,
) -> str:
    """Resolve a session id using only standard OpenAI-compatible inputs.

    Legacy custom fields still work for existing integrations. For standard OpenAI
    clients, the proxy uses the standard `user` field plus the first user message
    as the conversation key. If `user` is absent, it fingerprints the message
    history, which is the standard OpenAI mechanism for conversation continuity.
    """
    if isinstance(legacy_session_id, str) and legacy_session_id.strip():
        return legacy_session_id.strip()

    user = body.get("user")
    first_user = _first_user_message(messages)
    if isinstance(user, str) and user.strip():
        seed = _canonical_json(first_user)
        return f"user:{_hash_text(user)}:{_hash_text(seed)}"

    if first_user:
        return f"anon:{_hash_text(_canonical_json(first_user))}"

    return f"anon:{_hash_text(_canonical_json(messages))}"


def _pop_custom_session_fields(body: dict[str, Any]) -> tuple[Any, Any]:
    """Remove internal session fields so they are never forwarded downstream."""
    return body.pop("_session_id", None), body.pop("_session_state", None)


# Session -> stable backend slot mapping for prefix-cache reuse (review §1).
# A process-wide LRU map: slots are assigned lazily, and the least-recently-used
# entry is evicted past ``_SLOT_MAP_MAX`` so a long-running proxy serving many
# conversations does not grow this map without bound (review §4.10.1). Evicting an
# idle session's mapping is harmless — it is reassigned a slot on its next request.
_SLOT_MAP: OrderedDict[str, int] = OrderedDict()
_SLOT_MAP_MAX = 512
_SLOT_LOCK = threading.Lock()
_NEXT_SLOT = 0


def _slot_for_session(
    session_id: str, enabled: bool, total_slots: int = 0
) -> int | None:
    """Return a stable backend slot id for ``session_id`` or ``None``.

    Only assigns a slot when ``enabled`` is True (slot pinning is opt-in so
    non-llama.cpp backends stay OpenAI-transparent). The same session always
    maps to the same slot, which is what lets the backend reuse the whole
    conversation prefix across turns.

    ``enabled`` is resolved per request from live backend capabilities (see
    ``_slot_pinning_active``) so a session is never pinned to a slot the active
    device (e.g. NPU) does not have.

    ``total_slots`` is the backend's real slot count (from the ``/slots`` probe).
    The assigned id is clamped into ``[0, total_slots)`` so we never send an
    out-of-range ``id_slot``: llama.cpp mishandles the KV slot for an unknown id,
    which truncates long generations mid-stream and makes the backend fail to
    parse the (now-unterminated) tool-call arguments as JSON (a 500). When the
    backend exposes only a single slot (``total_slots <= 1``) pinning is skipped
    entirely -- there is nothing to gain from pinning a lone shared slot and it
    only risks cross-session KV collisions.
    """
    if not enabled or not session_id:
        return None
    if total_slots <= 1:
        # Single-slot (or unknown-count) server: pinning cannot isolate sessions
        # and colliding on the one slot corrupts concurrent long generations.
        return None
    with _SLOT_LOCK:
        slot = _SLOT_MAP.get(session_id)
        if slot is None:
            global _NEXT_SLOT
            slot = _NEXT_SLOT % total_slots
            _SLOT_MAP[session_id] = slot
            _NEXT_SLOT += 1
            while len(_SLOT_MAP) > _SLOT_MAP_MAX:
                _SLOT_MAP.popitem(last=False)
        else:
            _SLOT_MAP.move_to_end(session_id)
        return slot


def _slot_pinning_active(cfg: AppConfig, probe: Any | None) -> bool:
    """Resolve whether slot pinning should be used for THIS request.

    Precedence:
      1. Manual force-on: ``v050.slot_pinning_enabled`` always wins (operator
         explicitly opted in).
      2. Auto-detect: when ``v050.capability_autodetect`` is on and the live
         backend snapshot reports ``slot_pinning`` (the active device exposes
         ``/slots``, e.g. the GPU/llama.cpp runtime), enable it; when the active
         device has no slots (e.g. NPU), skip it.
      3. Otherwise off.

    Uses only the cached snapshot (no network on the request path); the snapshot
    is refreshed on its own TTL.
    """
    if cfg.v050.slot_pinning_enabled:
        return True
    if not cfg.v050.capability_autodetect or probe is None:
        return False
    caps = probe.cached()
    return bool(caps and caps.slot_pinning)


def _backend_total_slots(probe: Any | None) -> int:
    """Return the backend's reported slot count (0 when unknown).

    Read from the cached capability snapshot only (no network on the request
    path). Used to clamp assigned ``id_slot`` values into the valid range.
    """
    if probe is None:
        return 0
    caps = probe.cached()
    return int(caps.total_slots) if caps else 0


def _first_message_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the first message, for a one-time tokenizer calibration.

    The first message is usually the system prompt: large, stable, and
    representative of the model's real BPE, which makes it a good exact-count
    anchor. Handles both string content and OpenAI structured content parts.
    """
    if not messages:
        return ""
    content = messages[0].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(parts)
    return ""


def _normalize_response_choices(data: dict) -> list[dict]:
    """Pass backend response choices through unchanged.

    Qwen/llama.cpp can return explicit `reasoning_content` alongside `content`.
    The proxy must echo BOTH fields exactly as produced. Collapsing
    `reasoning_content` into `content` (the old behavior) made the client persist
    the reasoning as the assistant `content`, so the next turn's prefix differed
    from what the model actually generated — which broke prefix-cache reuse and
    MTP alignment (review §8.3). We therefore never mutate the message here.
    """
    return data.get("choices", [])


def _make_streaming_generator(
    body: dict,
    cfg: AppConfig,
    backend_client: LemonadeClient,
    optimizer: AgentContextOptimizer | None = None,
    id_slot: int | None = None,
    turn_start: float | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    prompt_hash: str | None = None,
) -> Any:
    """Create an async generator for SSE streaming using OpenAI SDK."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = body.get("model", cfg.server.llm_model)

    async def stream_generator() -> AsyncIterator[str]:
        initial_chunk = json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        })
        yield f"data: {initial_chunk}\n\n"

        cached_tokens: int | None = None
        backend_prompt_tokens: int | None = None
        backend_error = False
        try:
            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.1)
            max_tokens = body.get("max_tokens")
            request_kwargs = {
                key: value
                for key, value in body.items()
                if key not in {"messages", "model", "temperature", "max_tokens", "stream"}
            }

            # Request final-chunk usage (incl. cached_tokens) so the real prefix
            # cache outcome is reported even in streaming (review §8.1). Preserve
            # any caller-provided stream_options but force include_usage on.
            existing = request_kwargs.get("stream_options")
            if isinstance(existing, dict):
                existing = dict(existing)
                existing["include_usage"] = True
                request_kwargs["stream_options"] = existing
            else:
                request_kwargs["stream_options"] = {"include_usage": True}

            # Pin this session to a stable backend slot when slot pinning is on
            # (review §1). id_slot is a llama.cpp extension; it is only injected
            # when explicitly enabled so other backends stay OpenAI-transparent.
            if id_slot is not None:
                request_kwargs["id_slot"] = id_slot

            # P1.1 (cache guide DO #2): accumulate the assistant's content +
            # reasoning_content from the stream so we can remember the thinking
            # block the backend cached alongside this assistant message. Re-injected
            # on the next turn if the client stripped it (see optimizer.capture_thinking).
            _acc_content: list[str] = []
            _acc_reasoning: list[str] = []
            # Wall time of the first streamed CONTENT chunk, for a real TTFT
            # (review §4.12.1). reasoning/role-only chunks don't count.
            first_content_time: float | None = None

            # B1 expand continuation (review §4.1.2): with reversible compression on,
            # the model may call expand_content(handle) to retrieve a compressed tool
            # output's original. Tool-call chunks are BUFFERED (not streamed) so an
            # all-expand response can be fulfilled locally and the conversation
            # continued without the client seeing the internal round-trip. Content and
            # reasoning chunks stream immediately, so TTFT is unaffected for normal
            # (content) responses; tool-call chunks carry no content tokens, so
            # buffering them does not affect TTFT either.
            messages = list(messages)
            for _round in range(_MAX_EXPAND_ROUNDS):
                tool_calls_acc: dict[int, dict[str, str]] = {}
                buffered_tool_sse: list[str] = []
                saw_tool_calls = False
                tool_calls_finish = False
                # Authoritative usage captured from the backend's trailing chunk,
                # forwarded to the client as a final usage chunk (#3). Reset each
                # round so an expand-content re-query's usage supersedes the prior.
                final_usage: dict[str, Any] | None = None

                async for chunk in backend_client.chat_completions_stream(
                    messages=messages,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **request_kwargs,
                ):
                    # Extract fields from OpenAI SDK ChatCompletionChunk
                    delta = {}
                    finish_reason = None
                    tool_call_deltas = None

                    if hasattr(chunk, "choices") and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, "delta"):
                            d = choice.delta
                            if hasattr(d, "role") and d.role:
                                delta["role"] = d.role
                            if hasattr(d, "content") and d.content is not None:
                                delta["content"] = d.content
                                _acc_content.append(d.content)
                                if first_content_time is None:
                                    first_content_time = time.time()
                            if hasattr(d, "reasoning_content") and d.reasoning_content is not None:
                                delta["reasoning_content"] = d.reasoning_content
                                _acc_reasoning.append(d.reasoning_content)
                            if hasattr(d, "tool_calls") and d.tool_calls:
                                tool_call_deltas = d.tool_calls
                        if hasattr(choice, "finish_reason") and choice.finish_reason:
                            finish_reason = choice.finish_reason

                    # Some backends report usage (incl. cached_tokens) on the final
                    # chunk. Capture it so we can feed the real cache outcome to the
                    # hit-prediction model, AND forward it to the client as a final
                    # usage chunk so it sees completion_tokens / prompt_tokens (#3).
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        usage = chunk.usage
                        details = getattr(usage, "prompt_tokens_details", None)
                        details = details if isinstance(details, dict) else getattr(details, "__dict__", {})
                        cached_tokens = (
                            getattr(usage, "cache_hit_tokens", None)
                            or getattr(usage, "cached_tokens", None)
                            or details.get("cached_tokens")
                            if isinstance(details, dict)
                            else None
                        )
                        # Backend's true prompt token count for the optimized prompt
                        # we sent; used to calibrate the proxy's estimates (#6).
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                            backend_prompt_tokens = prompt_tokens
                        PROXY_METRICS.record_mtp_usage(
                            usage, request_id=request_id, session_id=session_id
                        )
                        # Serialize for forwarding to the client as the final usage
                        # chunk (empty choices + usage), OpenAI streaming format.
                        # Only forward a plain dict (a real backend usage object);
                        # skip non-dict stand-ins (e.g. test mocks) so json.dumps
                        # never raises mid-stream.
                        if isinstance(usage, dict):
                            final_usage = usage
                        else:
                            _dumped = usage.model_dump() if hasattr(usage, "model_dump") else None
                            final_usage = _dumped if isinstance(_dumped, dict) else None

                    # Buffer tool-call deltas and the tool_calls finish-reason chunk
                    # (replayed to the client only if the call is NOT an expand the
                    # proxy fulfils internally).
                    if tool_call_deltas is not None or finish_reason == "tool_calls":
                        if tool_call_deltas:
                            saw_tool_calls = True
                            for tc_delta in tool_call_deltas:
                                idx = getattr(tc_delta, "index", 0) or 0
                                acc = tool_calls_acc.setdefault(
                                    idx, {"id": "", "name": "", "arguments": ""}
                                )
                                if getattr(tc_delta, "id", None):
                                    acc["id"] = tc_delta.id
                                fn = getattr(tc_delta, "function", None)
                                if fn is not None:
                                    if getattr(fn, "name", None):
                                        acc["name"] = fn.name
                                    if getattr(fn, "arguments", None):
                                        acc["arguments"] += fn.arguments
                            delta["tool_calls"] = [
                                {
                                    "index": getattr(tc, "index", 0) or 0,
                                    "id": getattr(tc, "id", None),
                                    "type": "function",
                                    "function": {
                                        "name": getattr(getattr(tc, "function", None), "name", None),
                                        "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                                    },
                                }
                                for tc in tool_call_deltas
                            ]
                        if finish_reason == "tool_calls":
                            tool_calls_finish = True
                        sse_chunk = json.dumps({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }],
                        })
                        buffered_tool_sse.append(f"data: {sse_chunk}\n\n")
                        continue

                    if not delta and finish_reason is None:
                        continue

                    sse_chunk = json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }],
                    })
                    yield f"data: {sse_chunk}\n\n"

                    # Do NOT break on finish_reason. With stream_options.include_usage=True
                    # the backend emits the authoritative usage chunk (incl. cached_tokens)
                    # *after* the finish_reason chunk. Breaking here would skip the real
                    # prefix-cache outcome, so we let the loop run until the stream ends
                    # and capture usage on the trailing chunk below.

                # The inner stream ended. If it was an all-expand_content tool-call
                # response, fulfil it locally and re-query the backend; otherwise
                # replay any buffered (non-expand) tool-call chunks and finish.
                if saw_tool_calls and tool_calls_finish and tool_calls_acc:
                    assistant_msg = {
                        "role": "assistant",
                        "content": "".join(_acc_content) or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]},
                            }
                            for _idx, tc in sorted(tool_calls_acc.items())
                        ],
                    }
                    expand_results = _expand_tool_results(optimizer, assistant_msg)
                    if expand_results is not None:
                        messages.append(assistant_msg)
                        messages.extend(expand_results)
                        _acc_content = []
                        _acc_reasoning = []
                        continue  # re-query the backend with the fulfilled expand
                for sse in buffered_tool_sse:
                    yield sse
                # Forward the authoritative usage as the final chunk (OpenAI
                # streaming format: empty choices + usage) so the client sees
                # completion_tokens / prompt_tokens. Emitted once, after any
                # expand-content continuation resolves (#3).
                if final_usage is not None:
                    usage_chunk = json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [],
                        "usage": final_usage,
                    })
                    yield f"data: {usage_chunk}\n\n"
                break

        except (APIStatusError, APIError) as e:
            # Backend failed while streaming (e.g. HTTP 500 when the model's
            # tool-call arguments were truncated by max_tokens and llama.cpp
            # could not parse the unterminated JSON). Degrade gracefully: emit a
            # well-formed OpenAI error object + a terminating stop chunk so the
            # client sees a valid, closed stream instead of a broken connection.
            status = getattr(e, "status_code", None)
            logger.warning(
                "Backend error during streaming (status=%s): %s",
                status,
                type(e).__name__,
            )
            with suppress(Exception):
                PROXY_METRICS.record_backend_error(session_id)
            backend_error = True
            error_payload = json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }],
                "error": {
                    "message": f"Backend error while streaming: {type(e).__name__}",
                    "type": "backend_error",
                    "code": status,
                },
            })
            yield f"data: {error_payload}\n\n"

        except Exception as e:
            logger.exception("Streaming error in chat completions")
            with suppress(Exception):
                PROXY_METRICS.record_backend_error(session_id)
            backend_error = True
            error_chunk = json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }],
                "error": {
                    "message": f"Stream interrupted: {type(e).__name__}",
                    "type": "proxy_error",
                    "code": None,
                },
            })
            yield f"data: {error_chunk}\n\n"

        # On a backend error we already recorded the error counter above and must
        # not also record a phantom "successful" turn or calibrate on partial
        # data. Still emit the [DONE] sentinel so the SSE stream closes cleanly.
        if not backend_error:
            if optimizer is not None:
                try:
                    optimizer.record_cache_outcome(cached_tokens)
                except Exception:
                    logger.debug("Failed to record streaming cache outcome", exc_info=True)

            # P1.1 (cache guide DO #2): remember the thinking block we just
            # observed for this assistant turn, so the next turn can re-inject it
            # if the client stripped reasoning_content (keeps the backend's cached
            # prefix byte-stable and avoids a forced re-prefill).
            if optimizer is not None:
                try:
                    _assistant_content = "".join(_acc_content)
                    _assistant_reasoning = "".join(_acc_reasoning)
                    if _assistant_content:
                        optimizer.capture_thinking(_assistant_content, _assistant_reasoning or None)
                except Exception:
                    logger.debug("Failed to capture thinking block", exc_info=True)

            # Aggregate process-wide metrics from the authoritative backend signal.
            # TTFT = first CONTENT chunk time minus request start (review §4.12.1);
            # latency_ms remains the full end-to-end turn time.
            _ttft_ms: float | None = None
            if turn_start is not None and first_content_time is not None:
                _ttft_ms = (first_content_time - turn_start) * 1000.0
            PROXY_METRICS.record_turn(
                session_id=session_id,
                cached_tokens=cached_tokens,
                prompt_tokens=(optimizer.last_optimized_token_count if optimizer is not None else None),
                saved_tokens=(optimizer.last_saved_token_count if optimizer is not None else None),
                latency_ms=((time.time() - turn_start) * 1000.0 if turn_start is not None else None),
                ttft_ms=_ttft_ms,
                completion_tokens=(
                    final_usage.get("completion_tokens")
                    if isinstance(final_usage, dict)
                    and isinstance(final_usage.get("completion_tokens"), int)
                    else None
                ),
                completion_duration_ms=(
                    ((time.time() - turn_start) * 1000.0 - _ttft_ms)
                    if turn_start is not None and _ttft_ms is not None
                    else None
                ),
                request_id=request_id,
                prompt_hash=prompt_hash,
                slot=id_slot,
            )
            if optimizer is not None:
                PROXY_METRICS.record_degradations(optimizer.last_degradation_counts)

            # Calibrate the proxy's token estimates against the backend's real
            # tokenizer (review §1/§9, priority fix #6). The backend reports its true
            # `prompt_tokens` for the optimized prompt we sent; the ratio between that
            # and our tiktoken estimate lets the budget be enforced on true token
            # counts instead of an estimate that diverges for code-heavy prompts.
            if optimizer is not None and isinstance(backend_prompt_tokens, int) and backend_prompt_tokens > 0:
                try:
                    # Calibrate against the OPTIMIZED prompt we actually sent
                    # (optimizer._last_optimized), not the raw incoming messages, so
                    # the ratio reflects the true backend/proxy token gap (#6).
                    proxy_estimated_msgs = getattr(optimizer, "_last_optimized", None) or messages
                    proxy_estimated = optimizer.token_counter.count_messages(proxy_estimated_msgs)
                    if proxy_estimated > 0:
                        optimizer.set_token_calibration(backend_prompt_tokens / proxy_estimated)
                    # B0.6: also calibrate the remote-path per-message overhead so
                    # budget enforcement uses the backend's true token count.
                    optimizer.calibrate_remote_overhead(backend_prompt_tokens, proxy_estimated_msgs)
                except Exception:
                    logger.debug("Streaming token calibration failed", exc_info=True)

            # HTTP response headers are already sent when streaming begins, so the
            # real cache-hit signal cannot be exposed as an X- header here. Emit it
            # as an SSE comment line instead (valid SSE, ignored by clients but
            # visible to tooling) so the streaming path also surfaces reuse (review §8.2).
            if cached_tokens is not None:
                yield f": X-Prefix-Cache-Hit-Tokens: {cached_tokens}\n\n"

            # C8 (review §11.4): when front-eviction dropped complete turns this
            # turn, tell the client via an SSE comment so it knows history was
            # compacted. Valid SSE, ignored by clients but visible to tooling.
            evicted = getattr(optimizer, "last_evicted_turns", 0) or 0
            if evicted > 0:
                yield f": X-MOEPT-Context-Budget: evicted {evicted} turn(s)\n\n"

        yield "data: [DONE]\n\n"

    return stream_generator


# Max times the proxy will fulfil expand_content tool calls and re-query the backend
# within a single request, bounding the continuation loop (review §4.1.2 / B1).
_MAX_EXPAND_ROUNDS = 4


def _expand_tool_results(
    optimizer: AgentContextOptimizer | None, message: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Fulfil ``expand_content`` tool calls in an assistant message from the store.

    Returns the list of ``role=tool`` result messages to append when *every*
    tool_call in ``message`` is an ``expand_content`` call (so the proxy can satisfy
    them itself and re-query the backend). Returns ``None`` when the message has no
    tool_calls, or any tool_call is NOT ``expand_content`` (those must be returned to
    the client to fulfil). Never raises.
    """
    if optimizer is None:
        return None
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    results: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        if fn.get("name") != EXPAND_TOOL_NAME:
            return None
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        handle = str(args.get("handle") or "")
        try:
            original = optimizer.expand_content(handle)
        except Exception:  # pragma: no cover - defensive
            original = None
        results.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": (
                    original
                    if original is not None
                    else f"[content for handle {handle} is no longer available]"
                ),
            }
        )
    return results


async def _do_non_streaming(
    body: dict,
    session_state: str,
    cfg: AppConfig,
    backend_client: LemonadeClient,
    response_headers: dict[str, str] | None = None,
    optimization_error: str | None = None,
    optimizer: AgentContextOptimizer | None = None,
    id_slot: int | None = None,
    turn_start: float | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    prompt_hash: str | None = None,
) -> JSONResponse:
    """Execute non-streaming backend call using LemonadeClient (OpenAI SDK)."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = body.get("model", cfg.server.llm_model)

    try:
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.1)
        max_tokens = body.get("max_tokens")
        request_kwargs = {
            key: value
            for key, value in body.items()
            if key not in {"messages", "model", "temperature", "max_tokens", "stream"}
        }

        # Continuation loop for expand_content tool calls (review §4.1.2 / B1): with
        # reversible compression on, the model may call expand_content(handle) to
        # retrieve a compressed tool output's original. The proxy fulfils it from the
        # per-session ContentStore and re-queries the backend, transparently to the
        # client, until the model returns content (or a non-expand tool call). The
        # loop is bounded by _MAX_EXPAND_ROUNDS.
        backend_data: dict[str, Any] = {}
        for _round in range(_MAX_EXPAND_ROUNDS):
            response = await backend_client.chat_completions_create(
                messages=messages,
                model=model_name,
                temperature=temperature,
                stream=False,
                max_tokens=max_tokens,
                id_slot=id_slot,
                **request_kwargs,
            )
            # Convert OpenAI SDK ChatCompletion to dict format
            backend_data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            # Normalize choices for OpenAI compatibility
            _normalize_response_choices(backend_data)
            _choices = backend_data.get("choices", [])
            _assistant_msg = _choices[0].get("message", {}) if _choices else {}
            _expand_results = _expand_tool_results(optimizer, _assistant_msg)
            if _expand_results is None:
                break  # content, or a non-expand tool call -> return to the client
            # Fulfil expand_content locally and continue the conversation.
            messages.append(_assistant_msg)
            messages.extend(_expand_results)

        usage = backend_data.get("usage", {})

        # Log response details
        choices = backend_data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "")
            logger.debug(
                "Lemonade non-streaming response: content_len=%d, finish_reason=%s, usage=%s",
                len(content),
                finish_reason,
                usage,
            )
            if not content and finish_reason != "length":
                logger.warning(
                    "Lemonade returned empty content for %d messages (finish_reason=%s)",
                    len(messages),
                    finish_reason,
                )

        # Record cache hit for cache registry. Lemonade may expose cached tokens
        # either as top-level cache_hit_tokens or inside prompt_tokens_details.
        usage_dict = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
        PROXY_METRICS.record_mtp_usage(
            usage_dict, request_id=request_id, session_id=session_id
        )
        prompt_details = usage_dict.get("prompt_tokens_details", {}) or {}
        cache_hit_tokens = (
            usage_dict.get("cache_hit_tokens")
            or usage_dict.get("cached_tokens")
            or prompt_details.get("cached_tokens")
            or prompt_details.get("cache_hit_tokens")
        )
        if isinstance(cache_hit_tokens, int) and cache_hit_tokens > 0:
            from moeptimizer.cache_registry import get_cache_registry
            registry = get_cache_registry()
            registry.record_cache_hit(messages, cache_hit_tokens)

        # Feed the real backend cache outcome to the hit-prediction model so it
        # learns from actual reuse instead of a constant hit=True label.
        if optimizer is not None:
            try:
                optimizer.record_cache_outcome(cache_hit_tokens)
            except Exception:
                logger.debug("Failed to record cache outcome", exc_info=True)

        # Aggregate process-wide metrics from the authoritative backend signal.
        PROXY_METRICS.record_turn(
            session_id=session_id,
            cached_tokens=cache_hit_tokens if isinstance(cache_hit_tokens, int) else None,
            prompt_tokens=(optimizer.last_optimized_token_count if optimizer is not None else None),
            saved_tokens=(optimizer.last_saved_token_count if optimizer is not None else None),
            latency_ms=((time.time() - turn_start) * 1000.0 if turn_start is not None else None),
            completion_tokens=(
                usage_dict.get("completion_tokens")
                if isinstance(usage_dict.get("completion_tokens"), int)
                else None
            ),
            completion_duration_ms=(
                (time.time() - turn_start) * 1000.0 if turn_start is not None else None
            ),
            request_id=request_id,
            prompt_hash=prompt_hash,
            slot=id_slot,
        )
        if optimizer is not None:
            PROXY_METRICS.record_degradations(optimizer.last_degradation_counts)

        # Calibrate the proxy's token estimates against the backend's real
        # tokenizer (review §1/§9, priority fix #6). The backend reports its true
        # `prompt_tokens` for the optimized prompt we sent; the ratio between that
        # and our tiktoken estimate lets the budget be enforced on true token
        # counts instead of an estimate that diverges for code-heavy prompts.
        if optimizer is not None:
            try:
                backend_prompt_tokens = usage_dict.get("prompt_tokens")
                if isinstance(backend_prompt_tokens, int) and backend_prompt_tokens > 0:
                    # Calibrate against the OPTIMIZED prompt we actually sent
                    # (optimizer._last_optimized), not the raw incoming messages,
                    # so the ratio reflects the true backend/proxy token gap (#6).
                    proxy_estimated_msgs = getattr(optimizer, "_last_optimized", None) or messages
                    proxy_estimated = optimizer.token_counter.count_messages(proxy_estimated_msgs)
                    if proxy_estimated > 0:
                        optimizer.set_token_calibration(
                            backend_prompt_tokens / proxy_estimated
                        )
                    # B0.6: also calibrate the remote-path per-message overhead.
                    optimizer.calibrate_remote_overhead(backend_prompt_tokens, proxy_estimated_msgs)
            except Exception:
                logger.debug("Token calibration failed", exc_info=True)

        response_headers = dict(response_headers or {})
        if isinstance(cache_hit_tokens, int):
            response_headers["X-Prefix-Cache-Hit-Tokens"] = str(cache_hit_tokens)
        # C8 (review §11.4): surface eviction count on the non-streaming path too.
        evicted = getattr(optimizer, "last_evicted_turns", 0) or 0
        if evicted > 0:
            response_headers["X-MOEPT-Context-Budget"] = f"evicted {evicted} turn(s)"

        return JSONResponse(
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": backend_data.get("choices", []),
                "usage": usage,
            },
            headers={
                **dict(response_headers or {}),
            },
        )

    except (APIStatusError, APIError) as e:
        # Backend returned an error (e.g. HTTP 500 for a truncated tool call).
        # Surface it as a well-formed OpenAI error object, preserving the backend
        # status code where available, and record the backend_error metric.
        status = getattr(e, "status_code", None) or 502
        logger.warning(
            "Backend error in non-streaming completion (status=%s): %s",
            status,
            type(e).__name__,
        )
        with suppress(Exception):
            PROXY_METRICS.record_backend_error(session_id)
        response_headers = {}
        if optimization_error:
            response_headers["X-Optimization-Error"] = optimization_error
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": f"Backend error: {type(e).__name__}",
                    "type": "backend_error",
                    "param": None,
                    "code": getattr(e, "status_code", None),
                }
            },
            headers=response_headers,
        )

    except Exception as e:
        logger.exception("Non-streaming chat completion error")
        with suppress(Exception):
            PROXY_METRICS.record_backend_error(session_id)
        response_headers = {}
        if optimization_error:
            response_headers["X-Optimization-Error"] = optimization_error
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Internal error: {type(e).__name__}: {e}",
                    "type": "api_error",
                    "param": None,
                    "code": None,
                }
            },
            headers=response_headers,
        )


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or get_config()
    # Layer the selected quality preset (quality/balanced/aggressive) onto the
    # agentic config before any optimizer is built (review03.md §10).
    from moeptimizer.config import apply_quality_profile

    apply_quality_profile(cfg)
    # Live, device-aware capability probe (NPU<->GPU aware). Detects slot
    # pinning, native MTP, exact remote tokenization, and the tokenizer id from
    # the backend's own metadata, refreshed on a TTL so capabilities follow the
    # active device. Constructed even when autodetect is off (it is only *used*
    # when cfg.v050.capability_autodetect is on) so tests can inspect it.
    from moeptimizer.backend_capabilities import BackendCapabilityProbe

    capability_probe = BackendCapabilityProbe(
        base_url=cfg.server.url,
        model=cfg.server.llm_model,
        api_key=cfg.server.llm_api_key,
        ttl_seconds=cfg.v050.capability_probe_ttl_seconds,
    )
    # Created before the SessionManager so the one initialized instance can be
    # injected into every per-session optimizer (review §4.8.1). It is initialized
    # in ``lifespan`` below; optimizers are built lazily on first request, after
    # initialization, so the shared instance is ready by the time ranking runs.
    embedding_service = EmbeddingService()
    session_manager = SessionManager(
        config=cfg,
        capability_probe=capability_probe,
        embedding_service=embedding_service,
    )
    backend_client = LemonadeClient(
        base_url=cfg.server.url,
        api_key=cfg.server.llm_api_key,
        timeout=cfg.server.timeout,
        native_mtp_passthrough=cfg.v050.native_mtp_passthrough,
    )
    # DISABLED (review §4.2.5): OutputShaper clamps max_tokens / reasoning_effort
    # and injects a "be terse" system instruction — i.e. it tunes RESPONSE
    # verbosity, which violates the project's hard constraint ("the proxy compacts
    # ONLY the input context; the backend/model fully controls response size") and
    # cache_preservation_guide DONT #2 (varying generation params mid-session). It
    # also biased the direct-vs-proxy benchmark. The proxy must stay input-only;
    # response verbosity is addressed by better context fidelity, not output
    # clamping. Kept constructed (enabled=False) so shape_request is a no-op and the
    # wiring stays reversible behind an explicit, knowingly-constraint-violating edit.
    output_shaper = OutputShaper(enabled=False)
    embed_client = AsyncOpenAI(
        base_url=cfg.server.embed_url,
        api_key=cfg.server.embed_api_key,
        timeout=cfg.server.timeout,
    )

    # Lemonade exposes a standard OpenAI API. Do not enable proxy-level
    # speculative decoding wrappers here: the current backend does not expose
    # native MTP/speculative endpoints, and custom extra_body fields are not
    # part of the standard OpenAI chat-completions contract.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _OPTIMIZER_EXECUTOR
        # §9: dedicated bounded executor for the CPU-bound optimizer so it does
        # not compete with async-IO / embedding workers for default-pool threads.
        _OPTIMIZER_EXECUTOR = ThreadPoolExecutor(
            max_workers=max(1, cfg.agentic.optimizer_max_workers),
            thread_name_prefix="moept-optim",
        )

        # C9 (review §11.5): config hot-reload via SIGUSR2. The handler re-reads
        # AppConfig and swaps it into the SessionManager under its lock; existing
        # sessions keep their optimizer so in-flight requests never race a mid-turn
        # config change. Only registered when hot-reload is enabled and the platform
        # supports signals (not on Windows).
        _sigusr2_handler = None
        if cfg.agentic.config_hot_reload_enabled and hasattr(signal, "SIGUSR2"):

            def _handle_sigusr2(signum: int, frame: object) -> None:
                try:
                    session_manager.reload_config()
                    logger.info("Config hot-reloaded via SIGUSR2")
                except Exception as exc:  # Never crash the process from a signal
                    logger.warning("SIGUSR2 config reload failed: %s", exc)

            try:
                signal.signal(signal.SIGUSR2, _handle_sigusr2)
                _sigusr2_handler = _handle_sigusr2
            except (ValueError, OSError) as exc:
                # Signal handling is unavailable in this runtime (e.g. non-main
                # thread / unsupported platform). The /v1/config/reload endpoint
                # still works, so hot-reload is not fully lost.
                logger.debug("SIGUSR2 handler not registered: %s", exc)

        await embedding_service.initialize()
        # Live capability detection (review: NPU<->GPU aware). A single probe
        # reads the backend's own metadata (active device, /slots, native MTP,
        # exact /tokenize, tokenizer id) instead of guessing. This drives slot
        # pinning and MTP passthrough; tokenizer selection remains local-first and is refreshed on
        # a TTL per request so device hot-swaps are picked up without a restart.
        caps = None
        if cfg.v050.capability_autodetect:
            try:
                caps = await capability_probe.get(force=True)
                logger.info("Detected backend capabilities: %s", caps.summary())
            except Exception as exc:
                logger.warning("Capability detection failed: %s", exc)

        # Resolve MTP passthrough. Metadata (labels=['...','mtp'] or an active
        # speculative slot, or --spec-type ...mtp in the launch args) is a
        # reliable, non-invasive signal; prefer it over the chat probe.
        if not backend_client.native_mtp_passthrough:
            enabled_via_meta = bool(caps and caps.mtp)
            if enabled_via_meta:
                backend_client.enable_native_mtp_passthrough()
                logger.info(
                    "Backend metadata declares native MTP; enabling MTP "
                    "extra_body passthrough."
                )
            elif cfg.v050.native_mtp_autodetect:
                # Fallback: only chat-probe when metadata was inconclusive.
                try:
                    if await backend_client.detect_mtp_support():
                        backend_client.enable_native_mtp_passthrough()
                        logger.info(
                            "Backend chat-probe confirms native MTP; enabling "
                            "MTP extra_body passthrough."
                        )
                except Exception as exc:
                    logger.warning("MTP support auto-detection failed: %s", exc)

        # Keep the detected checkpoint for diagnostics, but do not turn it into
        # a network-capable tokenizer override. ``auto`` must remain local-first;
        # exact backend tokenization is handled by the capability probe.
        if (
            cfg.v050.capability_autodetect
            and cfg.server.tokenizer == "auto"
            and caps
            and caps.tokenizer_id
        ):
            app.state.detected_tokenizer_id = caps.tokenizer_id
            logger.info(
                "Detected backend tokenizer '%s'; keeping local-first tokenizer "
                "selection. Backend prompt_tokens still calibrates the residual.",
                caps.tokenizer_id,
            )

        logger.info(
            "Resolved native_mtp_passthrough=%s (autodetect=%s, capability_autodetect=%s); "
            "slot_pinning force=%s; the only functional speculative-decoding path is a "
            "backend with native MTP support.",
            backend_client.native_mtp_passthrough,
            cfg.v050.native_mtp_autodetect,
            cfg.v050.capability_autodetect,
            cfg.v050.slot_pinning_enabled,
        )
        # §4.10.3: reap expired sessions in the background so an idle proxy does not
        # retain optimizers past their timeout until the next request.
        session_manager.start_reaper()
        yield
        session_manager.stop_reaper()
        await embedding_service.close()
        await embed_client.close()
        await capability_probe.aclose()
        # C9: deregister the SIGUSR2 handler so a later process reusing this PID
        # does not inherit a dangling callback.
        if _sigusr2_handler is not None and hasattr(signal, "SIGUSR2"):
            with suppress(ValueError, OSError):
                signal.signal(signal.SIGUSR2, signal.SIG_DFL)
        # §9: shut down the dedicated optimizer executor after the server stops
        # accepting requests (in-flight requests have already completed by now).
        if _OPTIMIZER_EXECUTOR is not None:
            _OPTIMIZER_EXECUTOR.shutdown(wait=True)
            _OPTIMIZER_EXECUTOR = None

    app = FastAPI(
        title="Lemonade MoE Agentic Optimizer",
        description=(
            "Production-ready middleware for Qwen3.6-35B-A3B-MTP with agentic "
            "context management. Features: scratchpad compaction, thinking "
            "preservation, state-based RAG, LanceDB semantic index."
        ),
        lifespan=lifespan,
    )

    # Expose services for direct access by endpoints
    app.state.embedding_service = embedding_service
    app.state.backend_client = backend_client
    app.state.embed_client = embed_client
    app.state.capability_probe = capability_probe
    app.state.output_shaper = output_shaper

    @app.post("/v1/chat/completions")
    async def chat_completions_proxy(request: Request):
        """
        OpenAI-compatible chat completions proxy.

        Request schema (OpenAI):
          { model, messages, temperature, top_p, n, stream, stop, max_tokens,
            presence_penalty, frequency_penalty, logit_bias, user,
            tools, tool_choice, response_format }

        Conversation continuity:
          Existing `_session_id` / `_session_state` fields are still accepted,
          but standard OpenAI clients do not need them. The proxy derives the
          session key from the standard `user` field plus the first user message,
          or from a fingerprint of the message history when `user` is absent.
          Custom session fields are stripped before forwarding to Lemonade.
        """
        _turn_start = time.time()
        request_id = uuid.uuid4().hex[:16]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
            )

        messages = list(body.get("messages", []))
        if not messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid payload: no messages",
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": None,
                    }
                },
            )

        # Validate all non-assistant messages have 'content' field
        try:
            _validate_messages(messages)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": None,
                    }
                },
            )

        legacy_session_id, session_state = _pop_custom_session_fields(body)
        session_id = _resolve_session_id(body, messages, legacy_session_id)

        # Construct/fetch the optimizer OFF the event loop (review §4.9.1). On a new
        # session ``get_or_create`` builds ~25 components incl. disk I/O and a
        # (lru-cached) tokenizer load; running that inline stalled the whole loop and
        # blocked every concurrent request. The dedicated optimizer executor keeps it
        # off the loop and off the async-IO/embedding pools.
        loop = asyncio.get_running_loop()
        optimizer = await loop.run_in_executor(
            _OPTIMIZER_EXECUTOR, session_manager.get_or_create, session_id
        )

        if session_state:
            if session_id:
                await loop.run_in_executor(
                    _OPTIMIZER_EXECUTOR,
                    session_manager.load_state,
                    session_id,
                    session_state,
                )
                optimizer = await loop.run_in_executor(
                    _OPTIMIZER_EXECUTOR, session_manager.get_or_create, session_id
                )
            else:
                with suppress(Exception):
                    optimizer.load_session_state(session_state)

        # Capture the pristine original prompt text for the proxy-side
        # faithfulness computation BEFORE optimization mutates the messages (#7).
        # Only when diagnostics are requested (benchmarking), to avoid overhead.
        _diag_wanted = str(request.headers.get("X-MOEPT-Diagnostics", "")).strip().lower() in (
            "1", "true", "yes",
        )
        _original_prompt_text = _messages_to_text(messages) if _diag_wanted else ""

        optimized_messages = messages
        optimization_error: str | None = None
        token_count_before = optimizer.token_counter.get_timing_stats()
        optimization_started = time.perf_counter()
        try:
            # Run the (CPU-bound, synchronous) optimizer in a worker thread so the
            # asyncio event loop stays free for concurrent sessions. Previously the
            # optimizer ran inline on the event loop, so one long session blocked
            # all others (review §2/§4/§5). Use the dedicated optimizer executor
            # (review §9) so it does not compete with async-IO / embedding workers
            # for the default-pool threads.
            optimized_messages = await asyncio.get_running_loop().run_in_executor(
                _OPTIMIZER_EXECUTOR, optimizer.optimize_messages, messages
            )
        except Exception as e:
            logger.exception("Context optimization failed, falling back to recent-turn context")
            # Fold to header-safe ASCII: this string is surfaced verbatim in the
            # X-Optimization-Error response header, which must be latin-1-encodable.
            optimization_error = _header_safe(f"{type(e).__name__}: {e}")
            optimized_messages = _fallback_optimized_messages(messages, cfg.agentic.keep_full_steps)
        PROXY_METRICS.record_optimizer_duration(
            (time.perf_counter() - optimization_started) * 1000.0
        )
        token_count_after = optimizer.token_counter.get_timing_stats()
        PROXY_METRICS.record_token_count_duration(
            token_count_after[0] - token_count_before[0],
            token_count_after[1] - token_count_before[1],
        )

        # P1.2 (cache guide DO #5): pin the `tools` schema per session so the
        # backend's prefix cache (which includes the serialized tools array) is not
        # invalidated by client-side reordering. Re-emit the first-seen schema
        # verbatim on every turn. No-op when the client sent no tools.
        if isinstance(body.get("tools"), list):
            try:
                pinned = optimizer.pin_tools(body["tools"])
                if pinned is not None:
                    body["tools"] = pinned
            except Exception:
                logger.debug("tools pinning failed; forwarding client tools as-is", exc_info=True)

        # B1 (review §4.1.2): when reversible compression is on, advertise the
        # expand_content tool so the model can retrieve a compressed tool output's
        # original by handle. Appended AFTER pinning so the pinned client schema stays
        # byte-stable and the proxy's tool is added consistently every turn (keeping
        # the backend's serialized-tools cache valid). The proxy fulfils the call from
        # the per-session ContentStore (non-streaming continuation loop + streaming
        # generator) so the client never sees the internal expand round-trip.
        if cfg.agentic.reversible_compression_enabled and isinstance(body.get("tools"), list):
            body["tools"] = [*body["tools"], expand_content_tool()]

        # Dry-run mode (review §11 / P4a): when the X-MOEPT-Dry-Run header is set,
        # return the optimized prompt the proxy WOULD send to the backend plus a
        # token-savings diff and an estimated cache-hit signal — WITHOUT calling the
        # backend. This lets operators inspect what the proxy does to a request
        # (and how much it saves) without spending a single backend token. It is
        # purely observational and never mutates session state or metrics.
        dry_run = str(request.headers.get("X-MOEPT-Dry-Run", "")).strip().lower() in (
            "1",
            "true",
            "yes",
        ) or bool(body.get("_dry_run"))
        if dry_run:
            return _dry_run_response(optimizer, messages, optimized_messages, optimization_error)

        # Refresh live backend capabilities on their TTL (cheap: a no-op when the
        # cached snapshot is fresh). This is what lets slot pinning / MTP / remote
        # tokenization follow the active device when the backend hot-swaps between
        # NPU and GPU without restarting the proxy.
        if cfg.v050.capability_autodetect:
            with suppress(Exception):
                await capability_probe.get()

        # One-time exact-tokenizer calibration seed (review §1/§9, #6). Before this
        # session has ever seen a backend `prompt_tokens` response, anchor the
        # local-count->true-count ratio using the backend's own native /tokenize on
        # a representative sample (the system prompt / first message). This removes
        # turn-1 budget error even when the local tokenizer is the tiktoken
        # fallback. Runs at most once per session, only when the active device
        # exposes remote tokenization; best-effort and never blocks the request.
        if (
            cfg.v050.capability_autodetect
            and cfg.v050.remote_tokenize_enabled
            and not getattr(optimizer, "_calibration_seeded", False)
        ):
            caps = capability_probe.cached()
            if caps and caps.remote_tokenize:
                sample = _first_message_text(optimized_messages)
                if sample:
                    with suppress(Exception):
                        exact = await capability_probe.tokenize_count(sample)
                        if isinstance(exact, int) and exact > 0:
                            optimizer.seed_token_calibration(sample, exact)

        # Pin this session to a stable backend slot when slot pinning is active
        # for the CURRENT device (review §1). Resolved from live capabilities so a
        # session is never pinned to a slot an NPU device does not have. A stable
        # slot lets the backend reuse the whole conversation prefix across turns
        # instead of re-prefilling every turn.
        id_slot = _slot_for_session(
            session_id,
            _slot_pinning_active(cfg, capability_probe),
            _backend_total_slots(capability_probe),
        )

        # Debug logging for long contexts
        if len(optimized_messages) > 10:
            logger.info(
                "[Proxy] Turn with %d messages (original: %d), optimization_error=%s",
                len(optimized_messages),
                len(messages),
                optimization_error,
            )
            # Log message roles and content lengths
            for i, msg in enumerate(optimized_messages[:5]):
                logger.info(
                    "[Proxy] Message %d: role=%s, content_len=%d, preview=%s",
                    i,
                    msg.get("role"),
                    len(msg.get("content") or ""),
                    (msg.get("content") or "")[:100],
                )
            if len(optimized_messages) > 5:
                logger.info(
                    "[Proxy] ... and %d more messages",
                    len(optimized_messages) - 5,
                )

        # Ensure all non-assistant messages have 'content' for Lemonade compatibility.
        # The optimizer/compactor may produce tool_result or other non-assistant
        # messages that lack a content field from the original request.
        _ensure_content(optimized_messages)

        response_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Optimized-Prompt-Tokens": str(
                optimizer.last_optimized_token_count
                if optimizer.last_optimized_token_count is not None
                else optimizer.token_counter.count_messages(optimized_messages)
            ),
            "X-MOEPT-Request-Id": request_id,
        }

        # Expose the exact optimized prompt TEXT the proxy sends to the backend.
        # This is the proxy's one job (it compacts ONLY the input context), so
        # downstream tooling (e.g. the benchmark's prompt-faithfulness metric)
        # can measure how much of the original context survived compaction.
        # Gated to a sane size: full prompts can be huge, and the header is
        # only consumed by local benchmarking, not production clients.
        # Cheap pre-check before serializing (review §4.9.4): the rendered text is
        # never shorter than the raw string content, so if that alone exceeds the
        # header cap, skip the (potentially 100s-of-KB) build-then-discard entirely.
        _content_len = 0
        for _m in optimized_messages:
            _c = _m.get("content")
            if isinstance(_c, str):
                _content_len += len(_c)
        diagnostics_on = str(request.headers.get("X-MOEPT-Diagnostics", "")).strip().lower() in (
            "1", "true", "yes"
        )
        if diagnostics_on:
            # Proxy-side faithfulness metrics (#7): computed from the pristine
            # original text (captured before optimization) and the optimized text,
            # and emitted as bounded SCALAR headers so they survive even when the
            # full optimized prompt exceeds the 8 KB header cap — turns 3+ used to
            # drop to n/a because the text header was omitted. Mirrors the
            # benchmark's text-based metrics so the per-turn faithfulness chart is
            # populated for the whole run.
            _opt_text_raw = _messages_to_text(optimized_messages)
            _faith = _prompt_faithfulness(_original_prompt_text, _opt_text_raw)
            _source_recall = _prompt_source_token_recall(_original_prompt_text, _opt_text_raw)
            _evicted_recall = _evicted_content_recall(_original_prompt_text, _opt_text_raw)
            if _faith is not None:
                response_headers["X-MOEPT-Prompt-Faithfulness"] = f"{_faith:.6f}"
            if _source_recall is not None:
                response_headers["X-MOEPT-Source-Token-Recall"] = f"{_source_recall:.6f}"
            if _evicted_recall is not None:
                response_headers["X-MOEPT-Evicted-Content-Recall"] = f"{_evicted_recall:.6f}"
            # The full optimized prompt TEXT (for backward-compatible text-based
            # faithfulness in older benchmarks) stays bounded to 8 KB.
            _opt_text = _serialize_messages_text(optimized_messages) if _content_len <= 8000 else ""
            if _opt_text and len(_opt_text) <= 8000:
                # HTTP headers must be latin-1-encodable. Optimized prompt text can
                # contain unicode (em-dash, smart quotes, non-latin scripts from code
                # comments / fixture content); an un-sanitized value makes Starlette's
                # StreamingResponse header encoding raise UnicodeEncodeError -> HTTP
                # 500 for that turn. Fold non-latin-1 chars to safe ASCII substitutes
                # so the header (consumed only by local benchmarking) stays valid.
                response_headers["X-MOEPT-Optimized-Prompt-Text"] = _header_safe(_opt_text)
            else:
                response_headers["X-MOEPT-Diagnostics-Limit"] = (
                    "optimized prompt exceeds 8000-byte header limit"
                )

        # Dry-run / explain mode (review03.md §10): expose the exact optimized
        # prompt the proxy would send to the backend so operators can inspect
        # what changed. Opt-in per request via the X-MOEPT-Explain header, or
        # globally via agentic.explain_mode_enabled.
        explain_on = cfg.agentic.explain_mode_enabled or str(
            request.headers.get("X-MOEPT-Explain", "")
        ).strip().lower() in ("1", "true", "yes") or bool(body.get("_explain"))
        if explain_on:
            response_headers["X-MOEPT-Explain"] = "true"
            explain_payload = _explain_header_value(optimized_messages)
            if len(explain_payload) <= 8000:
                response_headers["X-MOEPT-Optimized-Messages"] = explain_payload
            else:
                response_headers["X-MOEPT-Explain-Limit"] = "optimized prompt exceeds 8000-byte header limit"

        # Serialize session state off the event loop (review §4.9.4): for long
        # agentic sessions get_session_state() json.dumps the whole store + runs
        # goal decomposition, which blocked concurrent requests when run inline.
        session_state = await asyncio.get_running_loop().run_in_executor(
            _OPTIMIZER_EXECUTOR, optimizer.get_session_state
        )
        existing_extra_body = body.get("extra_body")
        if existing_extra_body is not None and not isinstance(existing_extra_body, dict):
            logger.warning("Ignoring invalid extra_body value: %s", type(existing_extra_body).__name__)
            existing_extra_body = None
        backend_extra_body = optimizer.get_backend_extra_body(
            optimized_messages,
            existing_extra_body,
        )
        if backend_extra_body:
            body["extra_body"] = backend_extra_body
        if optimization_error:
            response_headers["X-Optimization-Error"] = optimization_error

        # Degradation vector (review §11 / P4b): surface any pipeline stages that
        # swallowed a failure this turn and fell back to a safe default. Operators
        # can use this to spot quality risk (e.g. RAG or code-block optimization
        # silently disabled) without scraping logs. Omitted entirely when the turn
        # ran clean so the header is only present on degraded responses.
        degradation = optimizer.last_degradation
        if degradation:
            response_headers["X-MOEPT-Optimization-Degraded"] = _header_safe("; ".join(degradation))

        body["model"] = cfg.server.llm_model
        # Scrub proxy-internal ``_*`` keys (e.g. the late-appended ``_volatile_turn``)
        # at the backend boundary so nothing internal reaches the model.
        body["messages"] = _scrub_internal_keys(optimized_messages)
        prompt_hash = hashlib.sha256(
            json.dumps(body["messages"], sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()[:16]
        response_headers["X-MOEPT-Prompt-Hash"] = prompt_hash
        if id_slot is not None:
            response_headers["X-MOEPT-Backend-Slot"] = str(id_slot)
        body.setdefault("temperature", 0.1)
        body.setdefault("stream", True)

        # Step: shape the backend request for output length (review §2.4 / P1).
        # Applies cache-safe system-prompt tail instruction + per-turn-class
        # max_tokens / reasoning_effort clamping. Does not touch the input path.
        output_shaper = getattr(app.state, "output_shaper", None)
        if output_shaper is not None:
            try:
                body = output_shaper.shape_request(body)
            except Exception as e:
                logger.debug("Output shaping failed: %s", e)

        is_streaming = body.get("stream", True)

        if is_streaming:
            # _make_streaming_generator returns a factory; invoke it to get the
            # async generator (an async iterator) that StreamingResponse expects.
            # Passing the bare factory function made Starlette do
            # `async for chunk in <function>` -> TypeError: 'function' object is
            # not iterable, which killed the stream and made clients see a
            # truncated response ("Response ended prematurely").
            return StreamingResponse(
                _make_streaming_generator(
                    body, cfg, backend_client, optimizer, id_slot, _turn_start, session_id,
                    request_id, prompt_hash,
                )(),
                media_type="text/event-stream",
                headers=response_headers,
            )
        else:
            return await _do_non_streaming(
                body, session_state, cfg, backend_client, response_headers, optimization_error,
                optimizer, id_slot, _turn_start, session_id, request_id, prompt_hash,
            )

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible models list endpoint."""
        return {
            "object": "list",
            "data": [
                {
                    "id": cfg.server.llm_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                },
                {
                    "id": cfg.server.embed_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "gemma",
                },
            ],
        }

    @app.get("/v1/metrics")
    async def proxy_metrics():
        """Process-wide proxy effectiveness metrics (review §11.1, fix #10).

        Surfaces whether the proxy is actually helping: real prefix-cache reuse
        (from the backend's authoritative ``cached_tokens``), token savings from
        optimization, cache-hit rate, and average latency. Not part of the OpenAI
        contract; purely observational for operators.
        """
        return {"object": "proxy.metrics", **PROXY_METRICS.snapshot()}

    @app.get("/v1/debug/requests")
    async def proxy_request_debug():
        """Return bounded per-request cache fingerprints for local diagnostics."""
        return {"object": "proxy.request_debug", "requests": PROXY_METRICS.request_traces()}

    @app.post("/v1/metrics/reset")
    async def proxy_metrics_reset():
        """Reset the process-wide proxy metrics counters."""
        PROXY_METRICS.reset()
        return {"object": "proxy.metrics", "status": "reset"}

    @app.get("/v1/metrics/ui")
    async def proxy_metrics_ui():
        """Live cache-reuse + token-savings dashboard (review §11 / P4c).

        A tiny self-contained HTML page (no framework, no external assets) that
        polls ``/v1/metrics`` and renders the proxy's effectiveness: aggregate
        prefix-cache reuse, token savings, average TTFT, and a per-session
        breakdown. Chart types follow the FT Visual Vocabulary (big-number
        "magic stat" for the headline, bar chart for per-session savings, line
        for cache-reuse ratio over the session list).
        """
        dashboard = _METRICS_DASHBOARD_HTML
        dashboard = dashboard.replace("__PROXY_VERSION__", html.escape(__version__))
        dashboard = dashboard.replace("__LLM_MODEL__", html.escape(cfg.server.llm_model))
        dashboard = dashboard.replace("__EMBED_MODEL__", html.escape(cfg.server.embed_model))
        return HTMLResponse(content=dashboard, status_code=200)

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request):
        """OpenAI-compatible embeddings endpoint (proxied to Lemonade)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
            )

        input_data = body.get("input")
        if input_data is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Missing required field: input",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": None,
                    }
                },
            )

        model = body.get("model", cfg.server.embed_model)

        if isinstance(input_data, str):
            input_list = [input_data]
        elif isinstance(input_data, list):
            if input_data and isinstance(input_data[0], dict):
                input_list = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in input_data
                ]
            else:
                input_list = [str(item) for item in input_data]
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid input format",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": None,
                    }
                },
            )

        embed_client = getattr(app.state, "embed_client", backend_client._client)
        # Bounded retry: the embed-gemma-300m-FLM backend occasionally returns an
        # empty response ("No embedding data received") on a transient NPU hiccup.
        # Try twice with a short backoff so a single hiccup does not drop the
        # embedding; only log — concisely, not a full traceback — if BOTH attempts
        # fail (a recovered retry logs at debug only).
        resp_dict: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(0.25)
            try:
                result = await embed_client.embeddings.create(
                    model=model,
                    input=input_list,
                )
                candidate = result.model_dump() if hasattr(result, "model_dump") else dict(result)
                if candidate.get("data"):
                    resp_dict = candidate
                    if attempt:
                        logger.debug("Embedding recovered on retry (model=%s)", model)
                    break
                # Empty data is a transient failure worth retrying.
                last_error = ValueError("No embedding data received")
            except Exception as e:
                last_error = e

        if resp_dict is None:
            logger.warning(
                "Embedding request failed after 2 attempts (model=%s, inputs=%d): %s",
                model,
                len(input_list),
                last_error,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Embedding error: {last_error}",
                        "type": "api_error",
                        "param": None,
                        "code": None,
                    }
                },
            )

        embeddings_data = resp_dict.get("data", [])
        usage = resp_dict.get("usage", {})
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": emb.get("embedding", []),
                    }
                    for i, emb in enumerate(embeddings_data)
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            },
        )

    @app.get("/v1/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "lemonade": "not_checked"}

    @app.post("/v1/agent/state")
    async def get_agent_state(request: Request):
        """Return current agent session state for persistence."""
        body = await request.json() if await request.body() else {}
        session_id = body.get("_session_id") or body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        optimizer = session_manager.get_or_create(session_id)
        progress = optimizer.progress_tracker.get_progress()

        return {
            "session_id": session_id,
            "session_state": optimizer.get_session_state(),
            "step_count": len(optimizer.store.steps),
            "goal": optimizer.store.get_goal().original_prompt if optimizer.store.get_goal() is not None else None,
            "progress": progress.to_dict(),
            "loop_warnings": [
                {"type": w.loop_type, "message": w.message}
                for w in optimizer.loop_detector.get_recent_warnings()
            ],
        }

    @app.post("/v1/agent/state/reset")
    async def reset_agent_state(request: Request):
        """Reset agent session state."""
        body = await request.json() if await request.body() else {}
        session_id = body.get("_session_id") or body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        session_manager.reset_session(session_id)
        return {"status": "ok", "message": "Agent state reset", "session_id": session_id}

    @app.get("/v1/agent/sessions")
    async def list_sessions():
        """List all active agent sessions."""
        sessions = session_manager.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}

    @app.get("/v1/agent/sessions/{session_id}/debug")
    async def session_debug(session_id: str):
        """Per-session debug dashboard (review §10, P4).

        Exposes the live-zone boundary (stable prefix vs. live zone), the real
        prefix-cache outcome, token savings, and the embedding circuit-breaker
        state so operators can see why a session is (or is not) reusing its KV
        cache and whether the embedding dependency is healthy. Read-only.
        """
        optimizer = session_manager.get_or_create(session_id)
        try:
            debug = optimizer.get_debug_info()
        except Exception as e:  # Never let a debug read crash the request path
            logger.debug("Failed to build session debug info: %s", e)
            debug = {"error": f"{type(e).__name__}: {e}"}
        debug["session_id"] = session_id
        debug["metrics"] = PROXY_METRICS.snapshot().get("sessions", {}).get(session_id)
        return {"object": "agent.session.debug", **debug}

    @app.get("/v1/agent/sessions/{session_id}/content/{handle}")
    async def get_retained_content(session_id: str, handle: str) -> dict[str, str]:
        """Retrieve the original of a reversibly-compressed tool output by handle
        (review §4.1.2 / Forward plan B1). The handle is embedded in the compressed
        placeholder; 404 if the content was evicted or never stored. A model-facing
        expand(id) tool that calls this is the planned follow-up."""
        optimizer = session_manager.get_or_create(session_id)
        content = optimizer.content_store.get(handle)
        if content is None:
            raise HTTPException(status_code=404, detail="handle not found or evicted")
        return {"object": "retained_content", "handle": handle, "content": content}

    @app.delete("/v1/agent/session/{session_id}")
    async def delete_session(session_id: str):
        """Delete an agent session."""
        deleted = session_manager.delete_session(session_id)
        return {"status": "ok" if deleted else "not_found", "session_id": session_id}

    @app.post("/v1/cache/clear")
    async def clear_caches():
        """Clear all caches."""
        embedding_service._embed_cache.clear()
        return {
            "status": "ok",
            "embed_cache_size": 0,
        }

    @app.post("/v1/config/reload")
    async def reload_config():
        """Hot-reload configuration from the environment without a restart (C9).

        Re-reads ``AppConfig`` (env / ``.env``) and applies the selected quality
        profile. New sessions pick up the new config immediately; existing sessions
        keep their optimizer so in-flight requests never race a mid-turn config
        change. Equivalent to sending ``SIGUSR2`` to the process. Useful for
        environments where signals are unavailable (containers, CI).
        """
        if not cfg.agentic.config_hot_reload_enabled:
            return JSONResponse(
                status_code=403,
                content={"status": "disabled", "detail": "config_hot_reload_enabled is false"},
            )
        try:
            new_config = session_manager.reload_config()
        except Exception as e:
            logger.warning("Config hot-reload failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "detail": f"{type(e).__name__}: {e}"},
            )
        return {
            "status": "ok",
            "quality_profile": new_config.agentic.quality_profile,
            "max_optimized_tokens": new_config.agentic.max_optimized_tokens,
            "rag_enabled": new_config.agentic.rag_enabled,
        }

    return app
