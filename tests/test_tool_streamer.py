"""Tests for tool output streaming."""


from moeptimizer.tool_streamer import (
    ToolOutputStreamer,
    get_tool_streamer,
)


class TestToolOutputStreamer:
    def test_small_output_not_streamed(self) -> None:
        """Small outputs are not streamed."""
        streamer = ToolOutputStreamer()
        small_output = "Line 1\nLine 2\nLine 3"
        assert not streamer.should_stream(small_output)

    def test_large_output_streamed(self) -> None:
        """Large outputs are candidates for streaming."""
        streamer = ToolOutputStreamer()
        large_output = "\n".join(f"Line {i}" for i in range(600))
        assert streamer.should_stream(large_output)

    def test_stream_output(self) -> None:
        """Stream output splits into content strings."""
        streamer = ToolOutputStreamer()
        large_output = "\n".join(f"Line {i}" for i in range(600))
        chunks = streamer.stream_output(large_output, "test_tool")
        assert len(chunks) > 1
        # All chunks should be strings
        assert all(isinstance(c, str) for c in chunks)

    def test_stream_code_blocks(self) -> None:
        """Code blocks are preserved during streaming."""
        streamer = ToolOutputStreamer()
        code_output = "```python\ndef foo():\n    pass\n```" * 100
        chunks = streamer.stream_output(code_output, "test_tool")
        # All code blocks should be preserved in the combined output
        combined = "".join(chunks)
        assert "```python" in combined

    def test_singleton(self) -> None:
        """Get tool streamer returns ToolOutputStreamer instance."""
        streamer1 = get_tool_streamer()
        streamer2 = get_tool_streamer()
        # Both should be ToolOutputStreamer instances
        assert isinstance(streamer1, ToolOutputStreamer)
        assert isinstance(streamer2, ToolOutputStreamer)
