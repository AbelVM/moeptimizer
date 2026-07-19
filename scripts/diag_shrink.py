"""Diagnostic: trace per-turn optimized token counts and per-stage deltas."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/fixtures")

from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer
from loader import build_fixture_agentic_tasks


def make_opt() -> AgentContextOptimizer:
    config = AppConfig()
    config.agentic.dynamic_budget_enabled = True
    config.agentic.budget_window_fraction = 0.025
    config.agentic.max_optimized_tokens = 12000
    config.agentic.max_optimized_chars = 48000
    config.agentic.max_context_growth_per_turn = 1500
    config.agentic.max_context_shrink_per_turn = 0
    config.agentic.keep_full_steps = 8
    config.agentic.quality_profile = "balanced"
    config.v050.cache_stable_mode = True
    config.v050.frozen_prefix_turns = 2
    config.v050.cache_stable_summary_enabled = True
    config.v050.hierarchical_summary_max_full_turns = 8

    class _Caps:
        max_context_window = 262144
        remote_tokenize = False

    class _Probe:
        def cached(self):
            return _Caps()

    return AgentContextOptimizer(config, capability_probe=_Probe())


def main() -> None:
    opt = make_opt()
    tc = opt.token_counter

    stages: dict[str, tuple[int, int]] = {}

    def wrap(name, orig):
        def wrapped(*a, **kw):
            msgs = a[0] if a else kw.get("messages")
            before = tc.count_messages(list(msgs)) if msgs is not None else -1
            res = orig(*a, **kw)
            after = tc.count_messages(list(res)) if res is not None else -1
            stages[name] = (before, after)
            return res

        return wrapped

    opt._proactive_trim = wrap("_proactive_trim", opt._proactive_trim)
    opt._sliding_window_trim = wrap("_sliding_window_trim", opt._sliding_window_trim)
    opt._trim_to_budget = wrap("_trim_to_budget", opt._trim_to_budget)
    opt.compactor.compact_messages = wrap("compact_messages", opt.compactor.compact_messages)
    opt.hierarchical_summarizer.summarize_turns_cache_stable = wrap(
        "summarize_turns_cache_stable", opt.hierarchical_summarizer.summarize_turns_cache_stable
    )
    opt.incremental_updater.update_context = wrap(
        "update_context", opt.incremental_updater.update_context
    )
    opt.context_compressor.compress = wrap(
        "context_compressor.compress", opt.context_compressor.compress
    )
    opt.context_canonicalizer.canonicalize = wrap(
        "context_canonicalizer.canonicalize", opt.context_canonicalizer.canonicalize
    )
    opt.context_template_matcher.apply_template = wrap(
        "apply_template", opt.context_template_matcher.apply_template
    )
    opt.thinking_preserver.process_messages = wrap(
        "process_messages", opt.thinking_preserver.process_messages
    )
    opt._strip_internal_flags = wrap("_strip_internal_flags", opt._strip_internal_flags)
    opt._append_volatile_context = wrap("_append_volatile_context", opt._append_volatile_context)
    opt.context_aligner.freeze_static_prefix = wrap(
        "freeze_static_prefix", opt.context_aligner.freeze_static_prefix
    )
    opt.token_aware_truncator.trim_messages_to_budget = wrap(
        "trim_messages_to_budget", opt.token_aware_truncator.trim_messages_to_budget
    )
    opt._merge_live_zone = wrap("_merge_live_zone", opt._merge_live_zone)
    opt._optimize_messages_locked = wrap(
        "_optimize_messages_locked", opt._optimize_messages_locked
    )

    # Track whether the incremental fast-path was taken.
    real_opt = opt._optimize_messages_locked

    def traced_optimize(messages, original_prompt=None):
        # We can't easily detect fast-path; instead patch optimize_messages wrapper.
        return real_opt(messages, original_prompt)

    opt._optimize_messages_locked = traced_optimize

    tasks = build_fixture_agentic_tasks(max_turns=30)
    conversation: list[dict] = [
        {"role": "system", "content": "You are a coding agent. Keep APIs stable."},
    ]
    prev = 0
    prev_stable = 0
    for n, turn in enumerate(tasks, start=1):
        conversation.extend(turn)
        optimized = opt.optimize_messages(list(conversation))
        tok = tc.count_messages(optimized)
        delta = tok - prev
        floor = opt._effective_shrink_floor()
        cap = opt._effective_shrink_cap()
        # Stable-prefix size = the leading block the backend can reuse (frozen +
        # append-only summary). A drop here == a prefix-cache break (the real
        # turn-11 cliff in the live benchmark: cached 3192 -> 882).
        stable = opt._live_zone_start
        stable_tok = tc.count_messages(optimized[:stable]) if stable else 0
        pflag = ""
        if n > 1 and stable_tok < prev_stable - 50:
            pflag = f"  <<< PREFIX BREAK (stable {prev_stable}->{stable_tok})"
        flag = ""
        if delta < -cap:
            flag = "  <<< SHRINK CLIFF (exceeds per-turn cap)"
        parts = " ".join(
            f"{k}:{v[0]}->{v[1]}" for k, v in stages.items() if v[0] != v[1]
        )
        print(
            f"turn {n:2d}: total={tok:5d} delta={delta:+6d} "
            f"stable={stable_tok:5d} cap={cap} floor={floor} | {parts}{pflag}{flag}"
        )
        prev = tok
        prev_stable = stable_tok


if __name__ == "__main__":
    main()
