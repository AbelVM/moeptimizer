"""Config sanity-check CLI (review03.md §10).

Validates the resolved :class:`AppConfig` and reports risky / contradictory
settings that would silently hurt prefix-cache reuse or response quality. Run
with ``moeptimizer-config-check`` (or ``python -m moeptimizer.config_check``).
Exits non-zero if any ERROR-level issue is found, so it can gate CI / deploy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from moeptimizer.config import AppConfig, apply_quality_profile, get_config


@dataclass(frozen=True)
class ConfigIssue:
    severity: str  # "ERROR" | "WARN" | "INFO"
    code: str
    message: str


def check_config(config: AppConfig) -> list[ConfigIssue]:
    """Return a list of config issues (errors, warnings, info notes)."""
    issues: list[ConfigIssue] = []
    a = config.agentic
    v = config.v050

    # ── Quality profile ────────────────────────────────────────────────────
    if a.quality_profile not in ("quality", "balanced", "aggressive"):
        issues.append(
            ConfigIssue(
                "WARN",
                "unknown_profile",
                f"quality_profile={a.quality_profile!r} is not a known preset "
                "(quality/balanced/aggressive); 'balanced' will be used.",
            )
        )

    # ── Budget / trim ordering ─────────────────────────────────────────────
    if a.max_optimized_tokens <= 0:
        issues.append(
            ConfigIssue("ERROR", "bad_budget", "max_optimized_tokens must be > 0.")
        )
    if not (0.0 < a.proactive_trim_ratio < a.compaction_trigger_ratio < 1.0):
        issues.append(
            ConfigIssue(
                "ERROR",
                "trim_order",
                "Require 0 < proactive_trim_ratio < compaction_trigger_ratio < 1 "
                f"(got proactive={a.proactive_trim_ratio}, compaction={a.compaction_trigger_ratio}).",
            )
        )
    if not (0.0 < a.eviction_low_water_ratio <= 1.0):
        issues.append(
            ConfigIssue(
                "ERROR",
                "low_water",
                "eviction_low_water_ratio must be in (0, 1] "
                f"(got {a.eviction_low_water_ratio}).",
            )
        )
    if a.keep_full_steps < 1:
        issues.append(
            ConfigIssue("WARN", "keep_full", "keep_full_steps < 1 keeps no full turns; quality may drop.")
        )

    # ── Prefix-cache killers (review03.md §0/§2.1) ─────────────────────────
    if v.hierarchical_summary_enabled:
        issues.append(
            ConfigIssue(
                "WARN",
                "summary_breaks_prefix",
                "hierarchical_summary_enabled mutates middle history and breaks the "
                "backend's contiguous prefix cache; reuse will drop.",
            )
        )
    if a.semantic_dedup_enabled:
        issues.append(
            ConfigIssue(
                "WARN",
                "dedup_breaks_prefix",
                "semantic_dedup_enabled removes middle-history messages and breaks the "
                "backend's contiguous prefix cache; reuse will drop.",
            )
        )
    if a.attention_sinks_enabled:
        issues.append(
            ConfigIssue(
                "WARN",
                "sinks_break_prefix",
                "attention_sinks_enabled injects model-visible markers into the prefix "
                "and breaks byte-stable cache reuse.",
            )
        )
    if a.reasoning_preseed_enabled:
        issues.append(
            ConfigIssue(
                "WARN",
                "preseed_mutates",
                "reasoning_preseed_enabled rewrites user messages; similarity to the "
                "direct baseline will fall (Grade D risk).",
            )
        )

    # ── Phantom / non-functional subsystems (review03.md §2.1) ─────────────
    if a.mtp_boundary_alignment_enabled:
        issues.append(
            ConfigIssue(
                "INFO",
                "mtp_noop",
                "mtp_boundary_alignment_enabled is a no-op for a client proxy (cannot "
                "touch MTP hidden state); it only pads the prompt with extra tokens.",
            )
        )
    if config.speculative.enabled and not v.native_mtp_passthrough:
        issues.append(
            ConfigIssue(
                "INFO",
                "speculative_stripped",
                "speculative.enabled sends MTP hints that are stripped before send "
                "unless v050.native_mtp_passthrough is on; effectively inert.",
            )
        )

    # ── Backend-compatibility flags ────────────────────────────────────────
    if v.enable_experimental_backend_hints:
        issues.append(
            ConfigIssue(
                "WARN",
                "experimental_hints",
                "v050.enable_experimental_backend_hints sends extra_body fields that "
                "unsupported backends may ignore or hang on.",
            )
        )
    if v.native_mtp_passthrough:
        issues.append(
            ConfigIssue(
                "INFO",
                "native_mtp",
                "v050.native_mtp_passthrough forwards MTP extra_body keys; only useful "
                "if the backend supports native speculative decoding.",
            )
        )
    if v.slot_pinning_enabled:
        issues.append(
            ConfigIssue(
                "INFO",
                "slot_pinning",
                "v050.slot_pinning_enabled pins sessions to a backend id_slot; only "
                "effective on llama.cpp/llama-server, not OpenAI-transparent backends.",
            )
        )

    # ── Tokenizer ──────────────────────────────────────────────────────────
    if a.quality_profile == "quality" and a.rag_enabled:
        # quality preset disables RAG; flag if an explicit override re-enabled it.
        issues.append(
            ConfigIssue(
                "INFO",
                "quality_rag",
                "quality profile normally disables RAG; rag_enabled=True was set "
                "explicitly and may reduce similarity to the direct baseline.",
            )
        )

    return issues


def format_issues(issues: list[ConfigIssue]) -> str:
    if not issues:
        return "OK: no config issues found."
    lines = []
    for issue in issues:
        lines.append(f"[{issue.severity}] {issue.code}: {issue.message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    config = get_config()
    apply_quality_profile(config)
    issues = check_config(config)
    print(format_issues(issues))
    has_error = any(i.severity == "ERROR" for i in issues)
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
