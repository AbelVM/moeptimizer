"""Configuration management for moeptimizer.

Loads settings from environment variables with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_file() -> str | None:
    """Find .env file, searching common locations.

    Priority: current dir > installed package location > user home.
    Returns the first existing .env path or None.
    """
    candidates = [
        Path.cwd() / ".env",                              # where user ran from
        Path(__file__).parent.parent.parent / ".env",     # dev install (src/)
        Path.home() / ".moeptimizer" / ".env",            # user-level fallback
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


_ENV_FILE = _resolve_env_file()


class ServerConfig(BaseModel):
    """Lemonade server connection settings."""

    url: str = Field(default="http://localhost:13305/api/v1")
    embed_url: str = Field(
        default="http://localhost:13305/api/v1",
        description="Base URL for the OpenAI-compatible embedding server.",
    )
    llm_model: str = Field(default="Qwen3.6-35B-A3B-MTP-GGUF")
    embed_model: str = Field(default="embed-gemma-300m-FLM")
    tokenizer: str = Field(
        default="auto",
        description=(
            "Tokenizer used for budget/counting (review §6 bug #1). One of: "
            "'auto' (try a local Qwen HF tokenizer, else fall back to tiktoken "
            "cl100k_base), a HuggingFace repo id or local directory/path to a "
            "tokenizer.json (loaded via transformers AutoTokenizer), or a tiktoken "
            "encoding name (e.g. 'cl100k_base'). NOTE: cl100k_base is GPT-4's BPE, "
            "not Qwen's, so it only approximates Qwen token counts; runtime "
            "calibration from the backend's real prompt_tokens corrects the ratio "
            "per turn regardless of this setting."
        ),
    )
    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds for long context conversations",
    )


class AgenticConfig(BaseModel):
    """Agentic loop tuning parameters for Qwen3.6-35B-A3B-MTP."""

    keep_full_steps: int = Field(
        default=3,
        description="Last N user-assistant pairs kept in full detail for immediate reasoning context. "
                    "Lower values improve token savings by evicting older complete turns from the top; "
                    "the optimizer never mutates middle-history content.",
    )
    archive_threshold: int = Field(
        default=3,
        description="Steps before this index are archived/compressed",
    )
    max_optimized_chars: int = Field(
        default=12000,
        description="Character fallback cap for optimized context (converted to ~3000 tokens)",
    )
    max_optimized_tokens: int = Field(
        default=3000,
        description="Hard cap on optimized context window (tokens). Takes precedence over max_optimized_chars if set.",
    )
    proactive_trim_ratio: float = Field(
        default=0.45,
        description="Ratio of max_optimized_tokens at which proactive top-only trimming starts.",
    )
    compaction_trigger_ratio: float = Field(
        default=0.75,
        description="Ratio of max_optimized_tokens at which compaction and compression start.",
    )
    thinking_protect_recent: int = Field(
        default=2,
        description="Keep full thinking for last N steps",
    )
    session_timeout: int = Field(
        default=3600,
        description="Session inactivity timeout in seconds",
    )
    use_token_budget: bool = Field(
        default=True,
        description="Use token-based budget enforcement instead of character-based",
    )
    fast_path_enabled: bool = Field(
        default=True,
        description="Bypass expensive transformations when the incoming context is already under the proactive budget.",
    )
    rag_enabled: bool = Field(
        default=True,
        description="Inject state-based RAG context only when it is enabled and useful for long/over-budget sessions.",
    )
    optimize_code_blocks: bool = Field(
        default=False,
        description="Run tree-sitter/NPU code-block optimization. Disabled by default to avoid proxy latency and preserve exact code.",
    )
    code_skeleton_enabled: bool = Field(
        default=True,
        description="Compress large code blocks to AST/line skeletons when proactive context pressure starts.",
    )
    semantic_dedup_enabled: bool = Field(
        default=False,
        description="Run embedding-based semantic deduplication when context pressure justifies it. Disabled by default because removing middle-history messages breaks KV-cache prefixes.",
    )
    attention_sinks_enabled: bool = Field(
        default=False,
        description="Inject model-visible attention-sink markers. Disabled by default to preserve exact prompts.",
    )
    static_layer_alignment_enabled: bool = Field(
        default=False,
        description="Pad the static layer to cache-block boundaries. Disabled by default to avoid adding tokens.",
    )
    reasoning_preseed_enabled: bool = Field(
        default=False,
        description="Inject reasoning scaffolding into user messages. Disabled by default to preserve direct-request semantics.",
    )
    tool_output_compression_enabled: bool = Field(
        default=True,
        description=(
            "Boundary-compress large tool/assistant outputs (logs, file dumps, "
            "RAG blobs) with cheap, lossless-ish transforms before they enter the "
            "stable prefix (headroom/snip-style). Applied once when a tool message "
            "first appears, so the compressed form is frozen into the prefix and "
            "the backend's prefix cache stays valid. Truncates oversized outputs "
            "(head+tail keep), collapses repeated stack frames, strips ANSI, and "
            "keeps code signatures. Disabled only if you need verbatim tool output."
        ),
    )
    tool_output_compression_max_chars: int = Field(
        default=4000,
        description="Tool/assistant outputs longer than this (chars) are boundary-compressed.",
    )
    eviction_low_water_ratio: float = Field(
        default=0.8,
        description=(
            "High/low watermark for budget eviction (review03.md §6/§9). Eviction "
            "is triggered when the evictable body exceeds the budget (high water), "
            "but then trims down to budget * this ratio (low water) in one batch. "
            "This keeps the oldest kept turn byte-stable across many subsequent "
            "turns instead of evicting one pair every over-budget turn, so the "
            "backend's native prefix cache (cached_tokens) is reused far more. Set "
            "to 1.0 to restore trim-to-exact-budget behavior."
        ),
    )
    mtp_boundary_alignment_enabled: bool = Field(
        default=False,
        description=(
            "Pad the final context to an MTP prediction boundary. NON-FUNCTIONAL: "
            "an OpenAI client proxy cannot read/write MTP hidden state, so this "
            "only does a no-op prompt pad (review03.md §2.1/§10). Disabled by "
            "default to avoid extra tokens."
        ),
    )
    immutable_prefix_enabled: bool = Field(
        default=True,
        description="Freeze the system prompt verbatim across turns so the backend's automatic "
                    "prefix cache can reuse it. The first user message is NOT frozen: it is "
                    "deterministically compressed and stays stable on its own, so freezing it would "
                    "undo compression. Volatile context (RAG, anchors, loop warnings) is only ever "
                    "appended to the last user turn.",
    )
    max_state_steps: int = Field(
        default=200,
        description="Maximum steps retained in AgentStateStore per session. Oldest archived steps are "
                    "pruned beyond this cap to bound memory growth over long agentic sessions.",
    )
    quality_profile: str = Field(
        default="balanced",
        description=(
            "Optimization preset that trades token savings against response fidelity "
            "(review03.md §10). One of: 'quality' (no summarization, RAG/anchor/code-skeleton "
            "off, only lossless boundary compression — maximizes similarity to the direct "
            "baseline), 'balanced' (current defaults), 'aggressive' (lower token cap, more "
            "top-only eviction, compaction starts earlier). Applied on top of any explicit "
            "field overrides, so individual fields can still be tuned."
        ),
    )
    explain_mode_enabled: bool = Field(
        default=False,
        description=(
            "Dry-run / explain mode (review03.md §10). When enabled, the proxy attaches "
            "the optimized prompt it would send to the backend as the `X-MOEPT-Optimized-Messages` "
            "response header (JSON) so operators can inspect exactly what the proxy changed. "
            "A single request can also opt in via the `X-MOEPT-Explain: true` request header, "
            "which works regardless of this flag. Off by default to avoid leaking full context "
            "in response headers."
        ),
    )


class CodeChunkingConfig(BaseModel):
    """Code chunking parameters."""

    chunk_max_chars: int = Field(default=1500)
    top_k_chunks: int = Field(default=5)
    min_chunk_score: float = Field(default=0.05)
    embedding_dim: int = Field(default=384)


class CacheConfig(BaseModel):
    """Cache settings."""

    embed_cache_max: int = Field(default=512)
    lancedb_path: str = Field(
        default=str(Path.home() / ".moeptimizer" / "lancedb"),
        description="Path to LanceDB data directory",
    )


class SpeculativeConfig(BaseModel):
    """Speculative decoding settings for MTP models.

    NOTE: these hints are sent as OpenAI extra_body keys and are stripped before
    forwarding unless `native_mtp_passthrough` is enabled (review03.md §2.1/§10).
    A client proxy cannot actually drive MTP; the only effective path is a
    backend that natively supports speculative decoding.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Enable MTP-aware speculative decoding hints. NON-FUNCTIONAL for a "
            "client proxy: hints are stripped before send unless the backend "
            "supports native MTP passthrough (review03.md §2.1/§10)."
        ),
    )
    mtp_lookahead: int = Field(
        default=4,
        description="Number of tokens to predict ahead with MTP heads",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for accepting speculative tokens",
    )


class V050Config(BaseModel):
    """v0.5.0 optimization settings."""

    # Static Prefix KV-Cache Reuse
    static_prefix_kv_enabled: bool = Field(
        default=True,
        description=(
            "Enable the static-prefix fast path. NOTE: this stores a TEXT memo of "
            "the system+first-user prefix, NOT real KV tensors -- a client-side "
            "OpenAI proxy cannot read the backend's KV cache. It only short-circuits "
            "the pipeline when the incoming prefix is byte-identical and already "
            "under budget; it does NOT reuse model KV (the backend's own prefix "
            "cache does that). Keep enabled for the latency win, but do not rely on "
            "it for KV reuse."
        ),
    )
    static_prefix_kv_max_entries: int = Field(
        default=64,
        description="Max entries in static prefix KV-cache",
    )

    # Token-Aware Truncation
    token_aware_truncation_enabled: bool = Field(
        default=True,
        description="Enable token-aware truncation with tiktoken",
    )

    # Chunk Fingerprinting & Reuse
    chunk_fingerprint_enabled: bool = Field(
        default=True,
        description="Enable chunk fingerprinting and reuse",
    )
    chunk_fingerprint_max_entries: int = Field(
        default=2048,
        description="Max entries in chunk fingerprint cache",
    )

    # Throttle cache_registry disk writes (review §10): the registry rewrites
    # the whole pickle on save, so we only persist every N turns instead of
    # every turn. The registry itself skips the write when nothing changed, but
    # the exact-context key changes each turn, so throttling bounds disk I/O.
    cache_registry_save_every: int = Field(
        default=10,
        description="Persist the cache registry to disk at most once every N turns (review §10).",
    )

    # Embedding Cache Invalidation & Batching
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding queries",
    )

    # Lightweight Hit-Prediction Model
    hit_prediction_enabled: bool = Field(
        default=True,
        description="Enable lightweight hit-prediction model",
    )
    hit_prediction_retrain_threshold: int = Field(
        default=50,
        description="Number of new samples before retraining",
    )

    # Hierarchical Summarization
    hierarchical_summary_enabled: bool = Field(
        default=False,
        description="Enable hierarchical summarization of old turns. Disabled by default because middle-history summaries break contiguous KV-cache prefixes.",
    )
    hierarchical_summary_max_full_turns: int = Field(
        default=5,
        description="Max recent turns to keep in full",
    )
    persist_state_to_disk: bool = Field(
        default=False,
        description=(
            "Persist optimizer state (hierarchical summaries, static-prefix KV "
            "memo, delta-encoder snapshots) to disk each turn. OFF by default: "
            "those subsystems are in-memory and the per-turn pickle writes add "
            "latency to the request path for no benefit on a single-process proxy "
            "(review03.md §8). Enable only if you need crash recovery / cross-process "
            "state. State is still kept in memory every turn regardless of this flag."
        ),
    )

    # Delta-Encoding of Code
    delta_encoding_enabled: bool = Field(
        default=True,
        description="Enable delta-encoding of code snapshots",
    )
    delta_encoding_max_snapshots: int = Field(
        default=100,
        description="Max code snapshots to keep",
    )

    enable_experimental_backend_hints: bool = Field(
        default=False,
        description="Send optional llama.cpp/MTP cache-control hints to the backend. Disabled by default because unsupported backends may ignore or hang on unknown extra_body fields.",
    )

    # Session -> backend slot pinning (review §1, priority fix #1)
    slot_pinning_enabled: bool = Field(
        default=False,
        description="Pin each session to a stable llama.cpp `id_slot` so the backend reuses the whole conversation prefix across turns. Disabled by default to stay OpenAI-transparent for non-llama.cpp backends. Requires the backend to support `id_slot` (llama.cpp/llama-server).",
    )

    # Cache-stable mode (review §1/§3/§7, priority fix #3): freeze a stable
    # prefix block verbatim so the backend's automatic prefix cache reuses it
    # across turns. The proxy's front-eviction (dropping old turns from the top)
    # shifts the serialized prefix every turn, which is why measured prefix-cache
    # reuse was ~0% even though the backend caches well. Freezing the early turns
    # trades a little token savings (they are kept uncompressed) for large
    # re-prefill savings, because the backend reuses the frozen prefix every turn.
    cache_stable_mode: bool = Field(
        default=True,
        description="Freeze a stable prefix block (system + first user + early turns) verbatim and never front-evict from it, so the backend reuses the KV cache across turns. Enabled by default because the proxy targets a prefix-caching backend (llama.cpp/Qwen3-MTP); disable for backends without prefix caching to maximize token savings.",
    )
    frozen_prefix_turns: int = Field(
        default=2,
        description="Number of early complete user-assistant turns (after the first user message) to freeze verbatim as part of the stable prefix when cache_stable_mode is enabled.",
    )

    # Native MTP speculative decoding passthrough (review §1, priority fix #2)
    native_mtp_passthrough: bool = Field(
        default=False,
        description="Do NOT strip MTP/speculative extra_body keys before forwarding to the backend, so a backend that supports native MTP speculative decoding (e.g. llama.cpp --speculative) receives them. Disabled by default because most backends reject unknown fields.",
    )
    native_mtp_autodetect: bool = Field(
        default=True,
        description="At startup, probe the backend for native MTP speculative decoding support and automatically enable `native_mtp_passthrough` when detected. Only takes effect when `native_mtp_passthrough` is False. Best-effort: a probe failure never blocks startup.",
    )

    # Async I/O for Heavy Stages
    async_io_enabled: bool = Field(
        default=True,
        description="Offload CPU/I/O-bound stages (tree-sitter compression, embedding ranking) to a "
                    "thread pool so the request/event-loop thread stays responsive under concurrent load.",
    )
    async_io_max_thread_workers: int = Field(
        default=4,
        description="Max thread workers for CPU-bound stages",
    )
    async_io_max_concurrency: int = Field(
        default=16,
        description="Max concurrent async tasks",
    )


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    port: int = Field(default=8080, description="Proxy server listen port")
    server: ServerConfig = Field(default_factory=ServerConfig)
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)
    code_chunking: CodeChunkingConfig = Field(default_factory=CodeChunkingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    v050: V050Config = Field(default_factory=V050Config)

    model_config = SettingsConfigDict(
        env_prefix="MOEPT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        **(
            {"env_file": _ENV_FILE}
            if _ENV_FILE is not None
            else {}
        ),
    )


def get_config() -> AppConfig:
    """Return the global application configuration."""
    return AppConfig()


# Quality-profile presets (review03.md §10). Each preset is a set of
# AgenticConfig field overrides layered on top of the loaded config. Explicit
# field values (env / .env) still win because the preset is applied first and
# individual overrides are then re-applied by pydantic's env loading order — to
# keep semantics simple, presets are applied at app-build time via
# `apply_quality_profile`, not during env parsing.
QUALITY_PROFILES: dict[str, dict[str, object]] = {
    "quality": {
        # Maximize fidelity to the direct baseline: no middle-history mutation,
        # only lossless boundary compression of oversized tool/assistant output.
        "hierarchical_summary_enabled": False,
        "rag_enabled": False,
        "reasoning_preseed_enabled": False,
        "code_skeleton_enabled": False,
        "semantic_dedup_enabled": False,
        "attention_sinks_enabled": False,
        "mtp_boundary_alignment_enabled": False,
        "keep_full_steps": 6,
        "max_optimized_tokens": 6000,
        "max_optimized_chars": 24000,
        "proactive_trim_ratio": 0.7,
        "compaction_trigger_ratio": 0.9,
    },
    "balanced": {
        # Current defaults; explicit here so the resolver is a single source.
        "hierarchical_summary_enabled": False,
        "rag_enabled": True,
        "reasoning_preseed_enabled": False,
        "code_skeleton_enabled": True,
        "semantic_dedup_enabled": False,
        "attention_sinks_enabled": False,
        "mtp_boundary_alignment_enabled": False,
        "keep_full_steps": 3,
        "max_optimized_tokens": 3000,
        "max_optimized_chars": 12000,
        "proactive_trim_ratio": 0.45,
        "compaction_trigger_ratio": 0.75,
    },
    "aggressive": {
        # Maximize token savings: lower cap, earlier compaction, more full steps
        # evicted from the top.
        "hierarchical_summary_enabled": False,
        "rag_enabled": True,
        "reasoning_preseed_enabled": False,
        "code_skeleton_enabled": True,
        "semantic_dedup_enabled": False,
        "attention_sinks_enabled": False,
        "mtp_boundary_alignment_enabled": False,
        "keep_full_steps": 2,
        "max_optimized_tokens": 2000,
        "max_optimized_chars": 8000,
        "proactive_trim_ratio": 0.35,
        "compaction_trigger_ratio": 0.6,
    },
}


def apply_quality_profile(config: AppConfig) -> AppConfig:
    """Layer the selected quality preset onto ``config`` in place.

    Each preset field is routed to whichever sub-config actually owns it
    (``agentic`` for loop-tuning fields, ``v050`` for fields like
    ``hierarchical_summary_enabled``). Returns ``config`` for chaining. Unknown
    profile names fall back to ``balanced`` with a warning so a typo never
    silently disables optimization.
    """
    profile = (config.agentic.quality_profile or "balanced").strip().lower()
    overrides = QUALITY_PROFILES.get(profile)
    if overrides is None:
        import logging

        logging.getLogger(__name__).warning(
            "Unknown quality_profile=%r; falling back to 'balanced'.", profile
        )
        profile = "balanced"
        overrides = QUALITY_PROFILES["balanced"]
        config.agentic.quality_profile = "balanced"
    # Route each override to the sub-config that owns the field. agentic is
    # checked first so loop-tuning fields land there; v050 owns the remainder
    # (e.g. hierarchical_summary_enabled).
    targets = (config.agentic, config.v050)
    for field, value in overrides.items():
        for target in targets:
            if hasattr(target, field):
                setattr(target, field, value)
                break
    return config
