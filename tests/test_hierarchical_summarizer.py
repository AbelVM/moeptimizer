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
            {"role": "user", "content": "Turn 1 implements the user registry module"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Turn 2 hardens the repository against bad input"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Turn 3 adds a configuration dataclass"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Turn 4 refactors the summarizer service"},
            {"role": "assistant", "content": "Resp 4"},
            {"role": "user", "content": "Turn 5 optimizes the summary step"},
            {"role": "assistant", "content": "Resp 5"},
            {"role": "user", "content": "Turn 6 adds lightweight in-memory metrics"},
            {"role": "assistant", "content": "Resp 6"},
            {"role": "user", "content": "Turn 7 emits a log event on each load"},
            {"role": "assistant", "content": "Resp 7"},
            {"role": "user", "content": "Turn 8 adds an argparse CLI entry point"},
            {"role": "assistant", "content": "Resp 8"},
        ]
        # Frozen prefix = system + first user + 2 turns (indices 0..4).
        # Dynamic layer = Turn2..Turn8 (7 turns). Batch folding fires once the
        # live window drifts fold_margin (=keep=3) past the keep window
        # (7 > 3+3), folds the oldest 4 turns into the summary and keeps the 3
        # most recent (Turn6..Turn8) verbatim.
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
        # the summary block: Turn5..Turn7 = messages[13:].
        assert result[6:] == messages[13:]

    def test_summarize_cache_stable_no_fold_within_margin(self) -> None:
        """No fold while the live window stays within keep + fold_margin.

        Folding mutates the prompt middle (one cache invalidation), so it must
        only fire once the live zone has drifted fold_margin turns past the
        keep window. Within the margin the messages are returned unchanged so
        the prompt stays a pure tail append.
        """
        summarizer = HierarchicalSummarizer(max_full_turns=3)  # margin defaults to keep=3
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            # 6 dynamic turns: live (6) <= keep (3) + margin (3) -> no fold.
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
            {"role": "user", "content": "Turn 6"},
            {"role": "assistant", "content": "Resp 6"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=5)
        assert result == messages
        assert not [m for m in result if m.get("_rolling_summary")]

    def test_summarize_cache_stable_retains_constraints(self) -> None:
        """The rolling summary retains the task's 'don't' constraints."""
        summarizer = HierarchicalSummarizer(max_full_turns=2)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Resp 0"},
            # OLD turns carrying the constraints that must survive in the
            # summary. 5 dynamic turns > keep (2) + margin (2) -> batch fold.
            {"role": "user", "content": "Please refactor, but do not change the public API."},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Also avoid renaming the database tables."},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Next, update the deployment scripts."},
            {"role": "assistant", "content": "Resp 2b"},
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
            # OLD turns carrying the code that must survive in the summary.
            # 5 dynamic turns > keep (2) + margin (2) -> batch fold.
            {"role": "user", "content": "Here is the helper:\n\n```python\n" + code + "\n```"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "Now document the helper in the readme."},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Then share it with the team."},
            {"role": "assistant", "content": "Resp 2b"},
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
                msgs.append(
                    {"role": "user", "content": f"Turn {i} implements the payment retry logic"}
                )
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
            blocks = [m for m in result if m.get("_rolling_summary")]
            text = "".join(b["content"] for b in blocks)
            # Append-only: the new summary must start with the previous one.
            if prev_text:
                assert text.startswith(prev_text), (
                    f"turn {n}: summary leading bytes changed (cache break)"
                )
            # Budget enforced at append time. Allow extra for per-message ROLLING_SUMMARY_MARKERs.
            assert len(text) <= 110 + (n * 35), f"turn {n}: summary over budget ({len(text)})"
            prev_text = text

        # Pinned facts always survive (byte-stable leading section, prepended at
        # build time and never rewritten).
        assert "API key abc123" in text

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
        # an OLD prose turn, a CODE turn, and RECENT prose turns. 5 dynamic
        # turns > keep (2) + margin (2) -> the oldest 3 fold in one batch.
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
            # One more old turn so the batch fold fires.
            {"role": "user", "content": "Also document the deployment steps for the release."},
            {"role": "assistant", "content": "Resp 2"},
            # Recent prose (kept verbatim in the live zone).
            {"role": "user", "content": "Now wire helper into the request handler and add logging."},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Then add a test for the handler."},
            {"role": "assistant", "content": "Resp 4"},
        ]
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=3)
        blocks = [m for m in result if m.get("_rolling_summary")]
        content = "".join(b["content"] for b in blocks)

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
            {"role": "user", "content": "Turn 1 implements the user registry module"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Turn 2 hardens the repository against bad input"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Turn 3 adds a configuration dataclass"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Turn 4 refactors the summarizer service"},
            {"role": "assistant", "content": "Resp 4"},
            {"role": "user", "content": "Turn 5 optimizes the summary step"},
            {"role": "assistant", "content": "Resp 5"},
            {"role": "user", "content": "Turn 6 adds lightweight in-memory metrics"},
            {"role": "assistant", "content": "Resp 6"},
            {"role": "user", "content": "Turn 7 emits a log event on each load"},
            {"role": "assistant", "content": "Resp 7"},
            {"role": "user", "content": "Turn 8 adds an argparse CLI entry point"},
            {"role": "assistant", "content": "Resp 8"},
        ]
        # Dynamic layer = Turn2..Turn8 (7 turns) > keep (3) + margin (3) ->
        # batch fold fires.
        result = summarizer.summarize_turns_cache_stable(messages, frozen_prefix_end=5)
        # Summary block is at index 5 (right after the 5-message frozen prefix).
        assert result[5].get("_summary_id")
        # It is NOT the last element (would be a trailing turn).
        assert result[5] is not result[-1]
        # The 3 most recent turns (Turn6..Turn8) follow the summary block.
        assert result[6:] == messages[13:]

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
                msgs.append(
                    {"role": "user", "content": f"Turn {i} implements the payment retry logic"}
                )
                msgs.append({"role": "assistant", "content": f"Resp {i}"})
            return msgs

        # Turn N: 7 dynamic turns, frozen_prefix_end=3. 7 > keep (3) + margin
        # (3) -> the first batch fold fires.
        r1 = summarizer.summarize_turns_cache_stable(conv(7), frozen_prefix_end=3)
        summary1 = next(m for m in r1 if m.get("_summary_id"))
        # Later turn: the live window has drifted past keep + margin again, so
        # a second batch fold APPENDS to the summary instead of rewriting it.
        r2 = summarizer.summarize_turns_cache_stable(conv(13), frozen_prefix_end=3)
        summary2 = next(m for m in r2 if m.get("_summary_id"))

        # The summary block is right after the frozen prefix in both.
        assert r1[3] is summary1
        assert r2[3] is summary2
        # The leading bytes of the summary (before the appended text) are stable.
        # summary2's content must START WITH summary1's content (append-only).
        assert summary2["content"].startswith(summary1["content"])
        # And the frozen prefix is byte-identical.
        assert [m.get("content") for m in r1[:3]] == [m.get("content") for m in r2[:3]]

    def test_batch_fold_keeps_prompt_append_only_between_folds(self) -> None:
        """Between batch folds the WHOLE emitted prompt is a pure tail append.

        The old per-turn sliding fold mutated the prompt middle every turn
        (the keep window's leading message was removed and the summary grew
        each turn), so the backend re-prefilled the entire live zone every
        turn — cached=0 from turn 14 in the 30-turn opencode log. With batch
        folding, every non-fold turn must emit exactly the previous prompt
        plus the new turn at the tail, so the backend reuses the whole
        previous prompt from its prefix cache.
        """
        summarizer = HierarchicalSummarizer(max_full_turns=2, fold_margin=2)

        def conv(n: int) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First user"},
                {"role": "assistant", "content": "Resp 0"},
            ]
            for i in range(n):
                msgs.append({"role": "user", "content": f"Turn {i} does some work"})
                msgs.append({"role": "assistant", "content": f"Resp {i}"})
            return msgs

        def serialize(msgs: list[dict[str, Any]]) -> str:
            return "\n".join(f"<|{m['role']}|>\n{m.get('content')}" for m in msgs)

        prev = ""
        folds = 0
        for n in range(1, 12):
            result = summarizer.summarize_turns_cache_stable(conv(n), frozen_prefix_end=3)
            blob = serialize(result)
            if prev and not blob.startswith(prev):
                # A fold turn: the only allowed divergence. The frozen prefix
                # must still stay byte-stable across the fold.
                folds += 1
                assert prev.startswith(serialize(result[:3])), (
                    f"turn {n}: frozen prefix changed on a fold turn"
                )
            prev = blob

        # keep=2, margin=2: folds fire when live > 4, i.e. at n=5, n=8, n=11.
        # Every other transition is a pure tail append.
        assert folds == 3

    def test_pressure_fold_measures_with_count_messages(self) -> None:
        """v0.7.26 quality regression: the pressure fold must measure the
        emitted size with ``count_messages`` (what the pipeline gates use),
        not content-only tokens.

        The scratchpad compactor is gated on ``count_messages``, which includes
        per-message overhead (e.g. tool-call payloads riding on empty-content
        assistant messages). The first version of the pressure fold summed
        ``len(content)`` only, undercounting by that overhead — so the fold
        stayed dormant while the compactor fired and dropped turns the summary
        had NOT captured (``evicted_content_recall`` collapsed to 0.51 in the
        16-turn benchmark). With a target that sits between the content-only
        size and the true ``count_messages`` size, the buggy measurement sees
        no pressure (no fold); the correct measurement folds.
        """

        class _OverheadCounter:
            # Per-message overhead count_messages sees but content-only misses.
            OVERHEAD = 40

            def count_tokens_precise(self, text: str) -> int:
                return max(1, len(text) // 4)

            def count_messages(self, messages: list[dict[str, Any]]) -> int:
                return sum(
                    max(1, len(str(m.get("content") or "")) // 4) + self.OVERHEAD
                    for m in messages
                )

        summarizer = HierarchicalSummarizer(max_full_turns=3)
        summarizer.set_token_counter(_OverheadCounter())

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user request for the task"},
        ]
        for i in range(10):
            messages.append(
                {"role": "user", "content": f"Step {i}: refactor the module and keep the API"}
            )
            messages.append({"role": "assistant", "content": f"Done with step {i}."})

        # Content-only size of the whole emitted list (system + first user +
        # live turns + the small summary block) vs the true count_messages
        # size; the target sits between them.
        content_only = sum(
            max(1, len(str(m.get("content") or "")) // 4) for m in messages
        )
        overhead = _OverheadCounter.OVERHEAD * len(messages)
        target = content_only + overhead // 2

        summarizer.summarize_turns_cache_stable(
            messages, frozen_prefix_end=2, pressure_target_tokens=target
        )
        assert summarizer._summarized_turn_count > 0, (
            "pressure fold did not fire: the emitted size was measured "
            "content-only instead of with count_messages, so the compactor "
            "would drop unsummarized turns (the v0.7.26 regression)"
        )
