"""Tests for hierarchical_summarizer module."""

from typing import Any

from moeptimizer.hierarchical_summarizer import (
    HierarchicalSummarizer,
    get_hierarchical_summarizer,
)


class TestHierarchicalSummarizer:
    def setup_method(self) -> None:
        self.summarizer = HierarchicalSummarizer(max_full_turns=3)

    def test_summarize_short(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.summarizer.summarize_turns(messages)
        assert len(result) == len(messages)

    def test_summarize_long(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Turn 4"},
            {"role": "assistant", "content": "Response 4"},
        ]
        result = self.summarizer.summarize_turns(messages)
        # Should have fewer messages after summarization
        assert len(result) < len(messages)
        # Should have a summary message
        summary_msgs = [m for m in result if m.get("_summary_id")]
        assert len(summary_msgs) == 1

    def test_expand_summary(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        result = self.summarizer.summarize_turns(messages)
        summary_msg = next(m for m in result if m.get("_summary_id"))
        expanded = self.summarizer.expand_summary(summary_msg)
        assert expanded["_summary_level"] == 2

    def test_get_summary(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.summarizer.summarize_turns(messages)
        summary_msg = [m for m in result if m.get("_summary_id")]
        if summary_msg:
            sid = summary_msg[0]["_summary_id"]
            stored = self.summarizer.get_summary(sid)
            assert stored is not None

    def test_clear(self) -> None:
        self.summarizer.summarize_turns([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        self.summarizer.clear()
        stats = self.summarizer.get_stats()
        assert stats["turns_summarized"] == 0

    def test_get_stats(self) -> None:
        messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Response 3"},
        ]
        self.summarizer.summarize_turns(messages)
        stats = self.summarizer.get_stats()
        assert "turns_summarized" in stats

    def test_global_instance(self) -> None:
        summarizer = get_hierarchical_summarizer()
        assert isinstance(summarizer, HierarchicalSummarizer)

    def test_summarize_cache_stable_short(self) -> None:
        """Short dynamic layer is returned unchanged (nothing to summarize)."""
        summarizer = HierarchicalSummarizer(max_full_turns=3)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Resp 2"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=2)
        assert result == messages

    def test_summarize_cache_stable_long_places_block_after_frozen(self) -> None:
        """Long dynamic layer gets a rolling summary block right after frozen prefix."""
        summarizer = HierarchicalSummarizer(max_full_turns=3)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Turn 4"},
            {"role": "assistant", "content": "Resp 4"},
            {"role": "user", "content": "Turn 5"},
            {"role": "assistant", "content": "Resp 5"},
        ]
        # Frozen prefix = system + first user + 2 turns (indices 0..4).
        # Dynamic layer = Turn2..Turn5 (4 turns, > max_full_turns=3) so the
        # oldest folds into the summary and the 3 most recent stay verbatim.
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=5)

        # Frozen prefix preserved verbatim at the front.
        assert result[0] == messages[0]
        assert result[1] == messages[1]
        assert result[2] == messages[2]
        assert result[3] == messages[3]
        assert result[4] == messages[4]

        # P0.5: the rolling summary block is placed IMMEDIATELY AFTER the frozen
        # prefix (not as a trailing turn). The backend prefix cache reuses the
        # LEADING bytes [frozen][summary], which are byte-stable across turns
        # (the summary only appends). The old trailing placement put [frozen]
        # [keep_recent] as the leading bytes, which change every turn and broke
        # prefix-cache reuse at turn 13.
        summary_block = result[5]
        assert summary_block.get("_summary_id")
        assert summary_block.get("_rolling_summary") is True
        assert summary_block["role"] == "user"
        assert "Context summary (rolling):" in summary_block["content"]

        # Most recent max_full_turns (3) turns retained in full, verbatim, AFTER
        # the summary block. Dynamic layer = Turn2..Turn5 (4 turns); the oldest
        # folds into the summary, so the 3 most recent (Turn3..Turn5) stay verbatim
        # at result[6:].
        assert result[6:] == messages[7:]

    def test_summarize_cache_stable_retains_constraints(self) -> None:
        """The rolling summary retains the task's 'don't' constraints."""
        summarizer = HierarchicalSummarizer(max_full_turns=2)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            # Two OLD turns carrying the constraints that must survive in the summary.
            {"role": "user", "content": "Please refactor, but do not change the public API."},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Also avoid renaming the database tables."},
            {"role": "assistant", "content": "Resp 2"},
            # Two RECENT turns kept in full (no constraint needed there).
            {"role": "user", "content": "Now add tests."},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Then run the suite."},
            {"role": "assistant", "content": "Resp 4"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=3)
        summary_block = next(m for m in result if m.get("_rolling_summary"))
        content = summary_block["content"]
        assert "do not change the public API" in content
        assert "avoid renaming the database tables" in content

    def test_summarize_cache_stable_preserves_code_blocks(self) -> None:
        """Evicted turns' fenced code survives in the rolling summary (P0.3 fix).

        The old _extract_constraints dropped fenced code entirely, so the model
        lost the task's code and has_code_proxy collapsed to 0. The summary must
        now carry the code block verbatim so the model can reproduce it.
        """
        summarizer = HierarchicalSummarizer(max_full_turns=2)
        code = "def process_items(items):\n    return [x for x in items if x]"
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            # Two OLD turns carrying the code that must survive in the summary.
            {"role": "user", "content": "Here is the helper:\n\n```python\n" + code + "\n```"},
            {"role": "assistant", "content": "Got it."},
            # Two RECENT turns kept in full.
            {"role": "user", "content": "Now use it."},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Then test it."},
            {"role": "assistant", "content": "Resp 4"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=3)
        summary_block = next(m for m in result if m.get("_rolling_summary"))
        content = summary_block["content"]
        assert "```python" in content
        assert code in content

    def test_summarize_cache_stable_append_only(self) -> None:
        """Across growing turns the rolling text only appends (leading bytes stable)."""
        summarizer = HierarchicalSummarizer(max_full_turns=2)

        def conv(n: int) -> list[dict]:
            msgs = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First user"},
                {"role": "assistant", "content": "Resp 0"},
            ]
            for i in range(n):
                msgs.append({"role": "user", "content": f"Turn {i}"})
                msgs.append({"role": "assistant", "content": f"Resp {i}"})
            return msgs

        r1 = summarizer.summarize_turns_cache_stable(conv(5), frozen_prefix_end=3)
        block1 = next(m for m in r1 if m.get("_rolling_summary"))
        text1 = block1["content"]

        r2 = summarizer.summarize_turns_cache_stable(conv(6), frozen_prefix_end=3)
        block2 = next(m for m in r2 if m.get("_rolling_summary"))
        text2 = block2["content"]

        # The second summary extends the first; the first's text is a prefix.
        assert text2.startswith(text1)
        assert len(text2) >= len(text1)
        # The summary id is stable across turns.
        assert block1["_summary_id"] == block2["_summary_id"]

    def test_rolling_summary_is_append_only_and_budget_capped(self) -> None:
        """Cache-stable summary: leading bytes never change; budget capped at append.

        The rolling summary is part of the STABLE PREFIX, so its leading bytes must
        stay byte-identical across turns (the backend reuses the cached KV for them).
        The v0.7.23 turn-11 cliff was caused by front-trimming the summary when it
        exceeded the token budget, which rewrote the leading bytes and dropped the
        cached prefix from 3192 -> 882 tokens. The fix enforces the budget at APPEND
        time (truncate the new text) and never rewrites existing content. A fake
        token counter (1 tok/char) makes the budget deterministic.
        """
        # Fake counter: 1 token per character so a small token cap is easy to hit.
        class _FakeCounter:
            def count_tokens_precise(self, text: str) -> int:
                return len(text)

        summarizer = HierarchicalSummarizer(
            max_full_turns=2, max_rolling_summary_tokens=110
        )
        summarizer.set_token_counter(_FakeCounter())

        # Seed pinned facts (byte-stable leading section, never evicted).
        summarizer.seed_original_request("API key abc123 base url http://x")

        def conv(n: int) -> list[dict]:
            msgs = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First user"},
                {"role": "assistant", "content": "Resp 0"},
            ]
            for i in range(n):
                msgs.append(
                    {"role": "user", "content": f"Turn {i} prose about the service"}
                )
                msgs.append({"role": "assistant", "content": f"Resp {i}"})
            return msgs

        # Build the summary incrementally across several turns. Each call must only
        # APPEND; the previously-cached leading bytes must remain identical.
        prev_text = ""
        for n in range(5, 12):
            result = summarizer.summarize_turns_cache_stable(
                conv(n), frozen_prefix_end=3
            )
            block = next(m for m in result if m.get("_rolling_summary"))
            text = summarizer._rolling_summary_text
            # Append-only: the new summary must start with the previous one.
            if prev_text:
                assert text.startswith(prev_text), (
                    f"turn {n}: summary leading bytes changed (cache break)"
                )
            # Budget enforced at append time: never exceeds the cap by much.
            assert len(text) <= 110 + 40, f"turn {n}: summary over budget ({len(text)})"
            prev_text = text

        # Pinned facts always survive (byte-stable leading section, prepended at
        # build time and never rewritten).
        assert "API key abc123" in block["content"]

    def test_rolling_summary_budget_keeps_code_over_prose(self) -> None:
        """High-value code is ordered before prose in extracted constraints (v0.7.15).

        The summary no longer front-trims (that broke the prefix cache). Instead
        :meth:`_extract_constraints` orders files/code before prose, so when the
        budget is enforced at append time the most valuable content sits at the
        front and survives. A fake token counter (1 tok/char) makes this visible.
        """
        # Fake counter: 1 token per character so a small token cap is easy to hit.
        class _FakeCounter:
            def count_tokens_precise(self, text: str) -> int:
                return len(text)

        summarizer = HierarchicalSummarizer(
            max_full_turns=2, max_rolling_summary_tokens=110
        )
        summarizer.set_token_counter(_FakeCounter())

        # Seed pinned facts (byte-stable leading section, never evicted).
        summarizer.seed_original_request("API key abc123 base url http://x")

        # Build a conversation whose folded turns produce >120 tokens of summary:
        # an OLD prose turn, a CODE turn, and a RECENT prose turn.
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            # Old prose (should be evicted first).
            {"role": "user", "content": "Old context: the project uses flask and a postgres db with redis cache layer."},
            {"role": "assistant", "content": "Resp 1"},
            # Code turn (must survive eviction).
            {"role": "user", "content": "Here is the helper:\n\n```python\ndef helper(x):\n    return x * 2\n```"},
            {"role": "assistant", "content": "Got it."},
            # Recent prose (kept: it is the newest folded turn).
            {"role": "user", "content": "Now wire helper into the request handler and add logging."},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Then add a test for the handler."},
            {"role": "assistant", "content": "Resp 4"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=3)
        block = next(m for m in result if m.get("_rolling_summary"))
        content = block["content"]

        # Pinned facts always survive (byte-stable leading section).
        assert "API key abc123" in content
        # Code block survives (it is ordered before prose in the extracted text).
        assert "```python" in content
        assert "def helper(x):" in content
        # The leading bytes of the summary are stable (no front-trim rewrite):
        # the code block appears before the old prose in the extracted text.
        assert content.index("```python") < content.index("postgres db with redis")

    def test_rolling_summary_budget_grows_with_turns(self) -> None:
        """Adaptive cap grows with folded turns, saturates at the ceiling (v0.7.17).

        The effective per-call cap is min(ceiling, floor + per_turn_growth *
        summarized_turn_count). A short session (few folded turns) gets a small
        cap; a long session grows toward the ceiling derived from the live backend
        window. The ceiling is set via set_rolling_summary_ceiling.
        """
        summarizer = HierarchicalSummarizer(max_full_turns=2, max_rolling_summary_tokens=200)
        # floor=200, ceiling defaults to 200 (== max_rolling_summary_tokens).
        assert summarizer._effective_summary_budget() == 200  # 0 folded turns -> floor

        # Raise the ceiling (as the optimizer does from the dynamic context budget).
        summarizer.set_rolling_summary_ceiling(2000)
        # With 0 folded turns the cap is still the floor.
        assert summarizer._effective_summary_budget() == 200
        # Simulate folded turns: cap grows by per_turn_growth (120) per turn.
        summarizer._summarized_turn_count = 5
        assert summarizer._effective_summary_budget() == 200 + 120 * 5  # 800
        summarizer._summarized_turn_count = 20
        # 200 + 120*20 = 2600, but ceiling is 2000 -> saturates.
        assert summarizer._effective_summary_budget() == 2000

    def test_rolling_summary_ceiling_clamped_to_floor(self) -> None:
        """A tiny derived ceiling never starves the summary below the floor."""
        summarizer = HierarchicalSummarizer(max_full_turns=2, max_rolling_summary_tokens=300)
        summarizer.set_rolling_summary_ceiling(50)  # below floor
        assert summarizer._rolling_summary_ceiling == 300  # clamped up to floor
        assert summarizer._effective_summary_budget() == 300

    def test_summary_block_placed_right_after_frozen_prefix(self) -> None:
        """P0.5: the summary block must sit immediately after the frozen prefix.

        The backend prefix cache reuses the LEADING bytes of the prompt. If the
        summary were a trailing turn, the leading bytes would be [frozen]
        [keep_recent], which change every turn (turns shift out of keep_recent
        into the folded set) and break prefix-cache reuse. Placing the summary
        right after the frozen prefix makes [frozen][summary] byte-stable.
        """
        summarizer = HierarchicalSummarizer(max_full_turns=3)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Turn 4"},
            {"role": "assistant", "content": "Resp 4"},
            {"role": "user", "content": "Turn 5"},
            {"role": "assistant", "content": "Resp 5"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=5)
        # Summary block is at index 5 (right after the 5-message frozen prefix).
        assert result[5].get("_summary_id")
        # It is NOT the last element (would be a trailing turn).
        assert result[5] is not result[-1]
        # keep_recent turns follow the summary block.
        assert result[6:] == messages[7:]

    def test_leading_prefix_byte_stable_across_turns(self) -> None:
        """P0.5: [frozen][summary] leading bytes are byte-stable as the summary grows.

        Simulates two consecutive turns where the summary appends new folded text.
        The leading bytes of the summary block (everything before the newly
        appended text) must be identical, so the backend reuses the KV for the
        frozen prefix + summary head instead of re-prefilling.
        """
        summarizer = HierarchicalSummarizer(max_full_turns=3)

        def conv(n_turns: int) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First user"},
                {"role": "assistant", "content": "Resp 0"},
            ]
            for i in range(n_turns):
                msgs.append({"role": "user", "content": f"Turn {i}"})
                msgs.append({"role": "assistant", "content": f"Resp {i}"})
            return msgs

        # Turn N: 6 dynamic turns (Turn0..Turn5), frozen_prefix_end=3.
        r1 = summarizer.summarize_turns_cache_stable(conv(6), frozen_prefix_end=3)
        summary1 = next(m for m in r1 if m.get("_summary_id"))
        # Turn N+1: one more turn added; the summary should APPEND, not rewrite.
        r2 = summarizer.summarize_turns_cache_stable(conv(7), frozen_prefix_end=3)
        summary2 = next(m for m in r2 if m.get("_summary_id"))

        # The summary block is right after the frozen prefix in both.
        assert r1[3] is summary1
        assert r2[3] is summary2
        # The leading bytes of the summary (before the appended text) are stable.
        # summary2's content must START WITH summary1's content (append-only).
        assert summary2["content"].startswith(summary1["content"])
        # And the frozen prefix is byte-identical.
        assert [m.get("content") for m in r1[:3]] == [m.get("content") for m in r2[:3]]
