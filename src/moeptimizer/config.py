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
    mtp_boundary_alignment_enabled: bool = Field(
        default=False,
        description="Pad the final context to an MTP prediction boundary. Disabled by default to avoid extra tokens.",
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
    """Speculative decoding settings for MTP models."""

    enabled: bool = Field(
        default=False,
        description="Enable MTP-aware speculative decoding",
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
        description="Enable static prefix KV-cache reuse",
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

    # Embedding Cache Invalidation & Batching
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding queries",
    )
    embedding_invalidation_enabled: bool = Field(
        default=True,
        description="Enable file mtime-based embedding invalidation",
    )

    # MTP-Head State Checkpointing
    mtp_checkpoint_enabled: bool = Field(
        default=True,
        description="Enable MTP-head state checkpointing",
    )
    mtp_checkpoint_max_entries: int = Field(
        default=256,
        description="Max entries in MTP head checkpoint cache",
    )

    # Parallel Embedding Lookup
    parallel_embed_workers: int = Field(
        default=8,
        description="Number of thread workers for parallel embedding",
    )

    # Segment-Wise Speculative Decoding
    segment_speculative_enabled: bool = Field(
        default=False,
        description="Enable segment-wise speculative decoding",
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

    # Template Selector
    template_selector_enabled: bool = Field(
        default=True,
        description="Enable template selector for cache optimization",
    )
    template_selector_exploration_rate: float = Field(
        default=0.1,
        description="Exploration rate for template selection",
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

    # Delta-Encoding of Code
    delta_encoding_enabled: bool = Field(
        default=True,
        description="Enable delta-encoding of code snapshots",
    )
    delta_encoding_max_snapshots: int = Field(
        default=100,
        description="Max code snapshots to keep",
    )

    # KV-Cache Warm-Up for MTP Heads
    kv_warmup_enabled: bool = Field(
        default=True,
        description="Enable KV-cache warm-up for MTP heads",
    )
    kv_warmup_max_entries: int = Field(
        default=32,
        description="Max warmup cache entries",
    )

    enable_experimental_backend_hints: bool = Field(
        default=False,
        description="Send optional llama.cpp/MTP cache-control hints to the backend. Disabled by default because unsupported backends may ignore or hang on unknown extra_body fields.",
    )

    # Async I/O for Heavy Stages
    async_io_enabled: bool = Field(
        default=True,
        description="Enable async I/O for heavy pipeline stages",
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
