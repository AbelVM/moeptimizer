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
"""

from __future__ import annotations

__version__ = "0.2.0"

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
from moeptimizer.code_chunking import (
    LANG_MAP,
    chunk_code_with_treesitter,
    chunk_text_fallback,
    deduplicate_chunks,
    detect_language_and_id,
)
from moeptimizer.expert_cache import (
    ExpertRoutingCache,
    get_expert_cache,
    hash_for_expert_routing,
)
from moeptimizer.optimizer import AgentContextOptimizer
from moeptimizer.prompt_templates import (
    PromptTemplateManager,
    classify_and_template,
    get_template_manager,
)
from moeptimizer.cache_aware_chunker import CacheAwareChunker, get_cache_aware_chunker
from moeptimizer.cache_registry import CacheKeyRegistry, get_cache_registry
from moeptimizer.context_aligner import ContextAligner, get_context_aligner
from moeptimizer.context_canonicalizer import ContextCanonicalizer, get_context_canonicalizer
from moeptimizer.context_compressor import ContextCompressor, get_context_compressor
from moeptimizer.context_template_matcher import ContextTemplateMatcher, get_context_template_matcher
from moeptimizer.dependency_orderer import DependencyOrderer, get_dependency_orderer
from moeptimizer.incremental_updater import IncrementalUpdater, get_incremental_updater
from moeptimizer.pattern_injector import PatternInjector, get_pattern_injector
from moeptimizer.selective_truncator import SelectiveTruncator, get_selective_truncator
from moeptimizer.symbol_index import SymbolIndex

__all__ = [
    "AgentContextOptimizer",
    "AttentionSinkManager",
    "CacheAwareChunker",
    "CacheKeyRegistry",
    "ContextAligner",
    "ContextCanonicalizer",
    "ContextCompressor",
    "ContextTemplateMatcher",
    "DependencyOrderer",
    "IncrementalUpdater",
    "PatternInjector",
    "SelectiveTruncator",
    "CONTEXT_BLOCK_SIZE",
    "ExpertRoutingCache",
    "LANG_MAP",
    "PromptTemplateManager",
    "SymbolIndex",
    "align_to_block_boundary",
    "apply_attention_sinks",
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
    "get_block_aligned_cache_key",
    "get_cache_aware_chunker",
    "get_cache_registry",
    "get_context_aligner",
    "get_context_canonicalizer",
    "get_context_compressor",
    "get_context_template_matcher",
    "get_dependency_orderer",
    "get_expert_cache",
    "get_incremental_updater",
    "get_pattern_injector",
    "get_selective_truncator",
    "get_template_manager",
    "hash_for_expert_routing",
]
