"""MoE Optimizer — Agentic context optimization middleware.

Transparent OpenAI API proxy that optimizes context for Qwen3.6-35B-A3B-MTP
and similar MoE + MTP models. Preserves reasoning continuity while compressing
stale context to prevent KV-cache thrashing.

Enhanced with:
- Static layer block alignment
- Multi-level cache key canonicalization
- Syntax-stable MTP prompt engineering
- Symbol index with fuzzy matching
- Dependency graph-aware context injection
- Hierarchical attention sink management
- Prompt template versioning
- Expert routing cache
- Speculative decoding support
- Static prefix KV-cache reuse
- Token-aware truncation with tiktoken
- Chunk fingerprinting & reuse
- Embedding cache invalidation & batching
- MTP-head state checkpointing
- Parallel embedding lookup
- Segment-wise speculative decoding
- Lightweight hit-prediction model
- Template selector
- Hierarchical summarization
- Delta-encoding of code
- KV-cache warm-up for MTP heads
- Async I/O for heavy stages
"""

from __future__ import annotations

__version__ = "0.5.1"

from moeptimizer.async_io_stage import AsyncIOStage, get_async_io_stage
from moeptimizer.attention_sink import AttentionSinkManager, apply_attention_sinks
from moeptimizer.cache import (
    CONTEXT_BLOCK_SIZE,
    align_to_block_boundary,
    cache_get,
    cache_key,
    cache_put,
    canonicalize_code_for_cache,
    canonicalize_prompt_for_cache,
    get_block_aligned_cache_key,
)
from moeptimizer.cache_aware_chunker import CacheAwareChunker, get_cache_aware_chunker
from moeptimizer.cache_registry import CacheKeyRegistry, get_cache_registry
from moeptimizer.chunk_fingerprint import ChunkFingerprintCache, get_chunk_fingerprint_cache
from moeptimizer.code_block_optimizer import (
    extract_code_blocks,
    has_code_blocks,
    optimize_code_in_text,
)
from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    chunk_text_fallback,
    deduplicate_chunks,
    detect_language_and_id,
)
from moeptimizer.compactor import ScratchpadCompactor
from moeptimizer.config import AppConfig, get_config
from moeptimizer.context_aligner import ContextAligner, get_context_aligner
from moeptimizer.context_canonicalizer import ContextCanonicalizer, get_context_canonicalizer
from moeptimizer.context_compressor import ContextCompressor, get_context_compressor
from moeptimizer.context_template_matcher import (
    ContextTemplateMatcher,
    get_context_template_matcher,
)
from moeptimizer.delta_encoder import CodeDeltaEncoder, get_delta_encoder
from moeptimizer.dependency_orderer import DependencyOrderer, get_dependency_orderer
from moeptimizer.embedding import EmbeddingService
from moeptimizer.embedding_cache_invalidation import (
    EmbeddingCacheWithInvalidation,
    get_embedding_cache_with_invalidation,
)
from moeptimizer.expert_cache import (
    ExpertRoutingCache,
    get_expert_cache,
    hash_for_expert_routing,
)
from moeptimizer.goal_decomposer import GoalDecomposer
from moeptimizer.hierarchical_index import get_hierarchical_index
from moeptimizer.hierarchical_summarizer import HierarchicalSummarizer, get_hierarchical_summarizer
from moeptimizer.hit_prediction_model import HitPredictionModel, get_hit_prediction_model
from moeptimizer.incremental_updater import IncrementalUpdater, get_incremental_updater
from moeptimizer.kv_cache_warmup import KVCacheWarmup, get_kv_cache_warmup
from moeptimizer.kv_slot_tracker import get_kv_slot_tracker
from moeptimizer.loop_detector import LoopDetector
from moeptimizer.models import AgentStep, LoopWarning
from moeptimizer.mtp_head_checkpoint import MTPHeadStateCheckpoint, get_mtp_head_checkpoint
from moeptimizer.mtp_speculative import MTPSpeculativeDecoder, build_mtp_speculative_body
from moeptimizer.mtp_state import MTPStateManager, get_mtp_state_manager
from moeptimizer.optimizer import AgentContextOptimizer
from moeptimizer.parallel_embedding_lookup import (
    ParallelEmbeddingLookup,
    get_parallel_embedding_lookup,
)
from moeptimizer.pattern_injector import PatternInjector, get_pattern_injector
from moeptimizer.progress_tracker import ProgressTracker
from moeptimizer.prompt_templates import (
    PromptTemplateManager,
    classify_and_template,
    get_template_manager,
)
from moeptimizer.segment_wise_speculative import (
    SegmentWiseSpeculativeDecoder,
    get_segment_wise_decoder,
)
from moeptimizer.selective_truncator import SelectiveTruncator, get_selective_truncator
from moeptimizer.semantic_dedup import SemanticDeduplicator, get_semantic_deduplicator
from moeptimizer.state_rag import StateBasedRAG
from moeptimizer.state_store import AgentStateStore
from moeptimizer.static_prefix_kv import StaticPrefixKVCache, get_static_prefix_kv_cache
from moeptimizer.summarize_old_turns import SummarizeOldTurns, get_summarize_old_turns
from moeptimizer.symbol_index import SymbolIndex
from moeptimizer.template_selector import TemplateSelector, get_template_selector
from moeptimizer.thinking_preserver import ThinkingPreserver
from moeptimizer.token_aware_truncator import TokenAwareTruncator
from moeptimizer.token_counter import TokenCounter
from moeptimizer.tool_streamer import get_tool_streamer

__all__ = [
    "CONTEXT_BLOCK_SIZE",
    "LANG_MAP",
    "AgentContextOptimizer",
    "AgentStateStore",
    "AgentStep",
    "AppConfig",
    "AsyncIOStage",
    "AttentionSinkManager",
    "CacheAwareChunker",
    "CacheKeyRegistry",
    "ChunkFingerprintCache",
    "CodeDeltaEncoder",
    "ContextAligner",
    "ContextCanonicalizer",
    "ContextCompressor",
    "ContextTemplateMatcher",
    "DependencyOrderer",
    "EmbeddingCacheWithInvalidation",
    "EmbeddingService",
    "ExpertRoutingCache",
    "GoalDecomposer",
    "HierarchicalSummarizer",
    "HitPredictionModel",
    "IncrementalUpdater",
    "KVCacheWarmup",
    "LoopDetector",
    "LoopWarning",
    "MTPHeadStateCheckpoint",
    "MTPSpeculativeDecoder",
    "MTPStateManager",
    "ParallelEmbeddingLookup",
    "PatternInjector",
    "ProgressTracker",
    "PromptTemplateManager",
    "ScratchpadCompactor",
    "SegmentWiseSpeculativeDecoder",
    "SelectiveTruncator",
    "SemanticDeduplicator",
    "StateBasedRAG",
    "StaticPrefixKVCache",
    "SummarizeOldTurns",
    "SymbolIndex",
    "TemplateSelector",
    "ThinkingPreserver",
    "TokenAwareTruncator",
    "TokenCounter",
    "align_to_block_boundary",
    "apply_attention_sinks",
    "build_mtp_speculative_body",
    "cache_get",
    "cache_key",
    "cache_put",
    "canonicalize_code_for_cache",
    "canonicalize_prompt_for_cache",
    "chunk_code_with_treesitter",
    "chunk_text_fallback",
    "classify_and_template",
    "deduplicate_chunks",
    "detect_language_and_id",
    "extract_code_blocks",
    "get_async_io_stage",
    "get_block_aligned_cache_key",
    "get_cache_aware_chunker",
    "get_cache_registry",
    "get_chunk_fingerprint_cache",
    "get_config",
    "get_context_aligner",
    "get_context_canonicalizer",
    "get_context_compressor",
    "get_context_template_matcher",
    "get_delta_encoder",
    "get_dependency_orderer",
    "get_embedding_cache_with_invalidation",
    "get_expert_cache",
    "get_hierarchical_index",
    "get_hierarchical_summarizer",
    "get_hit_prediction_model",
    "get_incremental_updater",
    "get_kv_cache_warmup",
    "get_kv_slot_tracker",
    "get_mtp_head_checkpoint",
    "get_mtp_state_manager",
    "get_parallel_embedding_lookup",
    "get_pattern_injector",
    "get_segment_wise_decoder",
    "get_selective_truncator",
    "get_semantic_deduplicator",
    "get_static_prefix_kv_cache",
    "get_summarize_old_turns",
    "get_template_manager",
    "get_template_selector",
    "get_tool_streamer",
    "has_code_blocks",
    "hash_for_expert_routing",
    "optimize_code_in_text",
]
