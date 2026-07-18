"""Tests for OutputShaper."""

from __future__ import annotations

from moeptimizer.output_shaper import OutputShaper, TurnClass


class TestOutputShaper:
    def test_disabled_returns_body_unchanged(self) -> None:
        shaper = OutputShaper(enabled=False)
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = shaper.shape_request(body)
        assert result == body

    def test_new_question_classification(self) -> None:
        shaper = OutputShaper()
        body = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 4096
        assert result["extra_body"]["reasoning_effort"] == "medium"

    def test_tool_result_classification(self) -> None:
        shaper = OutputShaper()
        body = {
            "messages": [
                {"role": "user", "content": "tool_result: 10 passed, 0 failed"}
            ]
        }
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 2048
        assert result["extra_body"]["reasoning_effort"] == "low"

    def test_error_classification(self) -> None:
        shaper = OutputShaper()
        body = {"messages": [{"role": "user", "content": "Error: connection refused"}]}
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 1024
        assert result["extra_body"]["reasoning_effort"] == "low"

    def test_continuation_classification(self) -> None:
        shaper = OutputShaper()
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Tell me more"},
            ]
        }
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 2048
        assert result["extra_body"]["reasoning_effort"] == "low"

    def test_terse_instruction_appended_to_system(self) -> None:
        shaper = OutputShaper()
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }
        result = shaper.shape_request(body)
        system_msg = result["messages"][0]
        assert "Be concise" in system_msg["content"]
        assert "You are helpful." in system_msg["content"]

    def test_terse_instruction_not_duplicated(self) -> None:
        shaper = OutputShaper()
        body = {
            "messages": [
                {"role": "system", "content": "Be concise. Answer directly."},
                {"role": "user", "content": "Hello"},
            ]
        }
        result = shaper.shape_request(body)
        system_msg = result["messages"][0]
        # The full terse instruction should appear exactly once.
        from moeptimizer.output_shaper import _TERSE_INSTRUCTION
        assert system_msg["content"].count(_TERSE_INSTRUCTION) == 1

    def test_existing_max_tokens_not_increased(self) -> None:
        shaper = OutputShaper()
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 100

    def test_existing_max_tokens_decreased_if_too_high(self) -> None:
        shaper = OutputShaper()
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10000,
        }
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 4096

    def test_custom_turn_class_overrides(self) -> None:
        shaper = OutputShaper(
            turn_class_max_tokens={TurnClass.NEW_QUESTION: 2048},
            turn_class_reasoning_effort={TurnClass.NEW_QUESTION: "low"},
        )
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = shaper.shape_request(body)
        assert result["max_tokens"] == 2048
        assert result["extra_body"]["reasoning_effort"] == "low"

    def test_no_system_message_no_crash(self) -> None:
        shaper = OutputShaper()
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = shaper.shape_request(body)
        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
