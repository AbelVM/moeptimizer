"""Tests for mtp_head_checkpoint module."""

import pickle

import pytest

from moeptimizer.mtp_head_checkpoint import MTPHeadStateCheckpoint, get_mtp_head_checkpoint


class TestMTPHeadStateCheckpoint:
    def setup_method(self) -> None:
        self.checkpoint = MTPHeadStateCheckpoint(max_checkpoints=10)

    def test_save_and_load(self) -> None:
        head_states = {0: b"state0", 1: b"state1", 2: b"state2"}
        key = self.checkpoint.save_head_states("def foo(x):", head_states)
        assert key != ""

        loaded = self.checkpoint.load_head_states("def foo(x):")
        assert loaded is not None
        assert loaded[0] == b"state0"
        assert loaded[1] == b"state1"
        assert loaded[2] == b"state2"

    def test_load_miss(self) -> None:
        result = self.checkpoint.load_head_states("def bar():")
        assert result is None

    def test_signature_key_stable(self) -> None:
        key1 = self.checkpoint._signature_key("def foo():")
        key2 = self.checkpoint._signature_key("def foo():")
        assert key1 == key2
        assert len(key1) == 32

    def test_signature_key_case_insensitive(self) -> None:
        key1 = self.checkpoint._signature_key("def Foo():")
        key2 = self.checkpoint._signature_key("def foo():")
        assert key1 == key2

    def test_has_checkpoint(self) -> None:
        self.checkpoint.save_head_states("def test():", {0: b"x"})
        assert self.checkpoint.has_checkpoint("def test():")
        assert not self.checkpoint.has_checkpoint("def other():")

    def test_invalidate(self) -> None:
        self.checkpoint.save_head_states("def test():", {0: b"x"})
        self.checkpoint.invalidate("def test():")
        assert not self.checkpoint.has_checkpoint("def test():")

    def test_clear(self) -> None:
        self.checkpoint.save_head_states("def a():", {0: b"a"})
        self.checkpoint.save_head_states("def b():", {0: b"b"})
        self.checkpoint.clear()
        assert self.checkpoint.get_stats()["saves"] == 0

    def test_lru_eviction(self) -> None:
        cp = MTPHeadStateCheckpoint(max_checkpoints=2)
        cp.save_head_states("def a():", {0: b"a"})
        cp.save_head_states("def b():", {0: b"b"})
        cp.save_head_states("def c():", {0: b"c"})
        assert cp.load_head_states("def a():") is None
        assert cp.load_head_states("def b():") is not None

    def test_get_or_create(self) -> None:
        calls = []

        def create(sig):
            calls.append(sig)
            return {0: f"state_for_{sig}".encode()}

        result1 = self.checkpoint.get_or_create("def new():", create)
        result2 = self.checkpoint.get_or_create("def new():", create)
        assert result1 == result2
        assert len(calls) == 1

    def test_get_stats(self) -> None:
        self.checkpoint.save_head_states("def a():", {0: b"a"})
        self.checkpoint.load_head_states("def a():")
        stats = self.checkpoint.get_stats()
        assert stats["saves"] == 1
        assert stats["loads"] == 1
        assert stats["hits"] == 1

    def test_global_instance(self) -> None:
        cp = get_mtp_head_checkpoint()
        assert isinstance(cp, MTPHeadStateCheckpoint)
