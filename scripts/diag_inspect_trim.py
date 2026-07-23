"""Inspect message list before token_aware_truncator at turn 16."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextlib import suppress
from moeptimizer import ROLLING_SUMMARY_MARKER
from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer

import benchmark as bench  # noqa: E402


class _FakeCaps:
    max_context_window = 262144
    remote_tokenize = False


class _FakeProbe:
    def cached(self):
        return _FakeCaps()

    def get(self):
        return _FakeCaps()


def make_optimizer() -> AgentContextOptimizer:
    config = AppConfig()
    config.agentic.quality_profile = "balanced"
    config.v050.static_prefix_kv_enabled = True
    config.v050.token_aware_truncation_enabled = True
    config.v050.hierarchical_summary_enabled = True
    config.v050.delta_encoding_enabled = True
    config.agentic.dynamic_budget_enabled = True
    opt = AgentContextOptimizer(config)
    opt._capability_probe = _FakeProbe()
    return opt


def main() -> None:
    num_turns = 16
    base_tasks = bench._OPENCODE_SCENARIO_TASKS
    base_tasks = bench._inject_drift_probe(base_tasks, num_turns)
    turn_exchanges: list[list[dict]] = [t for t in base_tasks if isinstance(t, list)]

    proxy_marker = "P{benchmark replay proxy session diag}\n"
    proxy_system_prompt = proxy_marker + bench.SYSTEM_PROMPT

    opt = make_optimizer()
    messages: list[dict] = [{"role": "system", "content": proxy_system_prompt}]

    for local_turn in range(num_turns):
        exchange = turn_exchanges[local_turn % len(turn_exchanges)]
        messages.extend(exchange)
        opt.optimize_messages(messages)

    messages.extend(turn_exchanges[16 % len(turn_exchanges)])

    # Patch _optimize_messages_locked to capture state before token_aware_truncator
    orig_optimize = opt._optimize_messages_locked

    def traced_optimize(msgs, original_prompt=None):
        # Run pipeline up to just before token_aware_truncator
        # We'll manually run the steps
        from moeptimizer.optimizer import AgentContextOptimizer

        # Copy the pipeline logic up to Step 12
        start_time = __import__('time').time()
        opt._last_degradation = []
        opt._last_evicted_turns = 0

        with suppress(Exception):
            msgs = opt._restore_thinking(msgs)

        live_zone_start = 0
        if opt._config.agentic.live_zone_compression_enabled:
            live_zone_start = opt._compute_live_zone_start(msgs)

        # ... skip incremental path for simplicity ...

        opt._ingest_messages(msgs)
        opt.store.set_max_steps(opt._dynamic_max_state_steps())

        if not opt.store.get_goal() and msgs:
            for msg in msgs:
                if msg.get("role") == "user":
                    goal_text = (msg.get("content") or "")[:500]
                    opt.store.set_goal(goal_text)
                    break

        if opt.hierarchical_summarizer is not None and opt._cache_stable_summary:
            first_user = next(
                (m.get("content") or "" for m in msgs if m.get("role") == "user"),
                "",
            )
            if first_user:
                opt.hierarchical_summarizer.seed_original_request(first_user)

        optimized = opt.thinking_preserver.process_messages(list(msgs))
        max_tokens = opt._effective_budget_tokens()
        proactive_threshold_tokens = int(max_tokens * opt._config.agentic.proactive_trim_ratio)
        compaction_threshold_tokens = int(max_tokens * opt._config.agentic.compaction_trigger_ratio)
        current_tokens = opt.token_counter.count_messages(optimized)
        total_tokens = current_tokens
        is_lean_context = total_tokens <= proactive_threshold_tokens

        fast_path = opt._maybe_fast_path(optimized, total_tokens, proactive_threshold_tokens)
        if fast_path is not None:
            return fast_path

        if opt.static_prefix_kv is not None:
            kv_data = opt.static_prefix_kv.get(optimized)
            if kv_data is not None:
                opt._last_static_prefix_hit = True
                if current_tokens <= proactive_threshold_tokens:
                    optimized = opt._strip_internal_flags(optimized)
                    return opt._finalize_optimized(optimized, msgs)

        # Step 7 pre-compaction: summary
        if (
            opt.hierarchical_summarizer is not None
            and opt._cache_stable_summary
            and opt._config.v050.cache_stable_mode
            and not is_lean_context
            and current_tokens > proactive_threshold_tokens
        ):
            opt.hierarchical_summarizer.set_rolling_summary_ceiling(
                int(max_tokens * opt._config.agentic.rolling_summary_budget_fraction)
            )
            frozen_end = opt.context_aligner.frozen_prefix_end(
                optimized, opt._config.v050.frozen_prefix_turns
            )
            optimized = opt.hierarchical_summarizer.summarize_turns_cache_stable(
                optimized, frozen_end
            )
            current_tokens = opt.token_counter.count_messages(optimized)

        # Step 7: compactor
        if current_tokens > compaction_threshold_tokens:
            shrink_floor = opt._effective_shrink_floor()
            optimized = opt.compactor.compact_messages(
                optimized, min_keep_tokens=shrink_floor
            )
            current_tokens = opt.token_counter.count_messages(optimized)

        # Step 11: proactive trim
        if total_tokens > proactive_threshold_tokens and not opt._prefix_drift:
            optimized = opt._proactive_trim(
                optimized, proactive_threshold_tokens, use_tokens=True,
                shrink_floor=opt._effective_shrink_floor(),
            )
            current_tokens = opt.token_counter.count_messages(optimized)

        # Step 11.5-11.7: boundary transforms
        optimized = opt._apply_boundary_transforms(optimized, live_zone_start)
        current_tokens = opt.token_counter.count_messages(optimized)

        # Step 11.8: sliding window
        if total_tokens > int(max_tokens * 0.8) and not opt._prefix_drift:
            optimized = opt._sliding_window_trim(
                optimized, use_tokens=True, shrink_floor=opt._effective_shrink_floor()
            )
            current_tokens = opt.token_counter.count_messages(optimized)

        # Step 12: trim to budget
        total_tokens = opt.calibrated_token_count(optimized)
        if total_tokens > max_tokens:
            optimized = opt._trim_to_budget(optimized, use_tokens=True)
            total_tokens = opt.calibrated_token_count(optimized)
            if total_tokens > max_tokens and opt.token_aware_truncator is not None:
                # CAPTURE STATE HERE
                print("=== BEFORE TOKEN_AWARE_TRUNCATOR ===")
                print(f"n_messages={len(optimized)}")
                frozen_end = opt.context_aligner.frozen_prefix_end(
                    optimized, opt._config.v050.frozen_prefix_turns
                )
                print(f"frozen_end={frozen_end}")
                for i, m in enumerate(optimized):
                    content = m.get("content") or ""
                    is_summary = content.startswith(ROLLING_SUMMARY_MARKER)
                    has_sid = bool(m.get("_summary_id"))
                    if is_summary or has_sid or i < frozen_end or i >= len(optimized) - 5:
                        print(f"  [{i:2d}] {m.get('role'):10s} is_summary={is_summary} has_sid={has_sid} content={content[:50]!r}")
                
                # Now run token_aware_truncator
                optimized = opt.token_aware_truncator.trim_messages_to_budget(
                    optimized,
                    max_tokens,
                )
                
                print("=== AFTER TOKEN_AWARE_TRUNCATOR ===")
                print(f"n_messages={len(optimized)}")
                for i, m in enumerate(optimized):
                    content = m.get("content") or ""
                    is_summary = content.startswith(ROLLING_SUMMARY_MARKER)
                    has_sid = bool(m.get("_summary_id"))
                    if is_summary or has_sid or i < 20 or i >= len(optimized) - 5:
                        print(f"  [{i:2d}] {m.get('role'):10s} is_summary={is_summary} has_sid={has_sid} content={content[:50]!r}")
                
                return opt._finalize_optimized(optimized, msgs)

        return opt._finalize_optimized(optimized, msgs)

    opt._optimize_messages_locked = traced_optimize
    result = opt.optimize_messages(messages)


if __name__ == "__main__":
    main()
