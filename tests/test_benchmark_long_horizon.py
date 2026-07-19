"""Unit tests for the long-horizon benchmark metrics.

Covers the three cross-turn signals added to scripts/benchmark.py:
  - fact recall (drift): _grade_fact_recall + _inject_drift_probe
  - contradiction rate:  _count_contradictions
  - context-window wall: _context_window_wall

These are pure functions; no backend / embedding calls are required except
where explicitly mocked.
"""

from unittest import mock

# scripts/ is on sys.path via tests/conftest.py
import benchmark as bm

# ---------------------------------------------------------------------------
# _inject_drift_probe
# ---------------------------------------------------------------------------


def test_inject_simple_tuple_prepends_anchor_and_appends_probe():
    tasks = [("user", "Task A"), ("user", "Task B"), ("user", "Task C")]
    out = bm._inject_drift_probe(tasks, 3)
    assert len(out) == 3
    # Turn 1 anchor prepended to the first user message.
    assert out[0][1].startswith(bm._DRIFT_PLANT)
    assert "Task A" in out[0][1]
    # Final turn is the recall probe (a list[dict] exchange).
    assert out[-1][0]["content"] == bm._DRIFT_PROBE


def test_inject_opencode_exchange_prepends_anchor():
    exch = [[{"role": "user", "content": "Build X"}, {"role": "assistant", "content": "ok"}]]
    out = bm._inject_drift_probe(exch, 5)
    assert len(out) == 5
    assert out[0][0]["content"].startswith(bm._DRIFT_PLANT)
    assert out[-1][0]["content"] == bm._DRIFT_PROBE


def test_inject_long_scenario_replaces_final_turn_with_probe():
    tasks = [[{"role": "user", "content": f"t{i}"}] for i in range(10)]
    out = bm._inject_drift_probe(tasks, 6)
    assert len(out) == 6
    assert out[0][0]["content"].startswith(bm._DRIFT_PLANT)
    assert out[-1][0]["content"] == bm._DRIFT_PROBE
    # Earlier turns are preserved verbatim.
    assert out[1][0]["content"] == "t1"


def test_inject_empty_tasks_is_noop():
    assert bm._inject_drift_probe([], 5) == []


# ---------------------------------------------------------------------------
# _count_contradictions
# ---------------------------------------------------------------------------


def test_contradiction_detects_negation_flip():
    turns = [
        "The database is Postgres. We use Python 3.11.",
        "The database is not Postgres.",  # explicit flip on shared subject
    ]
    assert bm._count_contradictions(turns) >= 1


def test_contradiction_ignores_consistent_statements():
    turns = [
        "We use Python 3.11.",
        "We target Python 3.11.",  # both affirmative, same subject -> not a flip
    ]
    assert bm._count_contradictions(turns) == 0


def test_contradiction_empty():
    assert bm._count_contradictions([]) == 0


# ---------------------------------------------------------------------------
# _context_window_wall
# ---------------------------------------------------------------------------


class _FakeTurn:
    def __init__(self, idx, quality):
        self.turn_index = idx
        self.quality = quality


def test_wall_finds_first_collapse():
    turns = [
        _FakeTurn(1, {"code_block_ratio": 1.0, "semantic_similarity": 0.8}),
        _FakeTurn(2, {"code_block_ratio": 0.3, "semantic_similarity": 0.5}),
    ]
    wall = bm._context_window_wall(turns)
    assert wall["proxy"] == 2
    assert wall["direct"] == 2


def test_wall_none_when_no_collapse():
    turns = [
        _FakeTurn(1, {"code_block_ratio": 1.0, "semantic_similarity": 0.8}),
        _FakeTurn(2, {"code_block_ratio": 0.9, "semantic_similarity": 0.7}),
    ]
    wall = bm._context_window_wall(turns)
    assert wall["proxy"] is None
    assert wall["direct"] is None


# ---------------------------------------------------------------------------
# _grade_fact_recall
# ---------------------------------------------------------------------------


def test_fact_recall_returns_none_on_empty_response():
    assert bm._grade_fact_recall("", bm._DRIFT_FACTS) is None


def test_fact_recall_grades_lexically_without_embedding():
    # Fact recall is graded primarily by lexical (normalized substring) matching
    # of each fact's answer tokens, so it must work with NO embedding model
    # available. We force the embedder down to prove the primary path is
    # embedding-independent (the old grader returned 0.0 for a verbatim recall
    # because whole-response embedding similarity was diluted by boilerplate).
    with mock.patch.object(bm, "_embed_text", side_effect=RuntimeError("no backend")):
        # Response states every fact's answer tokens -> full recall.
        response = (
            "The codename is ATLAS. We target Python 3.11. The database is Postgres. "
            "The max retry count is 3. The owning team is platform-infra."
        )
        score = bm._grade_fact_recall(response, bm._DRIFT_FACTS)
        assert score == 1.0

        # Response mentions only some facts -> partial recall.
        partial = bm._grade_fact_recall(
            "The codename is ATLAS. The database is Postgres.", bm._DRIFT_FACTS
        )
        assert partial == 0.4  # 2 of 5 facts

        # Response mentions none of the facts -> zero recall.
        score0 = bm._grade_fact_recall("The weather is sunny today.", bm._DRIFT_FACTS)
        assert score0 == 0.0

        # Empty response -> None (not measured), never a false zero.
        assert bm._grade_fact_recall("", bm._DRIFT_FACTS) is None


def test_fact_recall_uses_embedding_only_as_fallback():
    # When lexical finds nothing, the embedder is consulted as a soft fallback
    # for paraphrased recalls. With the embedder forced down, a non-lexical
    # (paraphrased) response stays at 0.0 rather than erroring.
    with mock.patch.object(bm, "_embed_text", side_effect=RuntimeError("no backend")):
        paraphrased = (
            "I recall we named the initiative ATLAS and chose Postgres for storage."
        )
        # "ATLAS" and "Postgres" are lexical matches; "Python 3.11", "retry 3",
        # "platform-infra" are not stated -> 0.4, not an error.
        assert bm._grade_fact_recall(paraphrased, bm._DRIFT_FACTS) == 0.4
