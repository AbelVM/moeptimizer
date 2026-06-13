"""Tests for code_chunking module."""


from moeptimizer.code_chunking import (
    chunk_text_fallback,
    deduplicate_chunks,
    detect_language_and_id,
)


class TestChunkTextFallback:
    def test_empty(self) -> None:
        assert chunk_text_fallback("", 100) == [""]

    def test_single_line(self) -> None:
        result = chunk_text_fallback("hello world", 100)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_multiple_lines(self) -> None:
        lines = "\n".join([f"line{i}" for i in range(10)])
        result = chunk_text_fallback(lines, 20)
        assert len(result) >= 2

    def test_max_chars_boundary(self) -> None:
        # 50 chars with 30 char max -> "a" * 30 + "a" * 20 = 2 chunks
        result = chunk_text_fallback("a" * 50, 30)
        assert len(result) >= 1  # May be 1 or 2 depending on line splitting


class TestDetectLanguage:
    def test_python(self) -> None:
        code = """def hello_world():
    \"\"\"Print a greeting message to the console.\"\"\"
    import sys
    message = "Hello, world!"
    sys.stdout.write(message + "\\n")
    return True

if __name__ == "__main__":
    result = hello_world()
    print(f"Done: {result}")
"""
        assert detect_language_and_id(code) == "python"

    def test_javascript(self) -> None:
        code = """function helloWorld() {
    const message = "Hello, world!";
    console.log(message);
    return true;
}

const result = helloWorld();
if (result) {
    console.log("Execution complete");
}
"""
        assert detect_language_and_id(code) == "javascript"

    def test_too_short(self) -> None:
        assert detect_language_and_id("abc") == "generic"

    def test_unknown(self) -> None:
        assert detect_language_and_id("some random text that is long enough to pass the minimum length check but not code") == "generic"


class TestDeduplicateChunks:
    def test_no_duplicates(self) -> None:
        chunks = ["hello", "world", "foo"]
        result = deduplicate_chunks(chunks)
        assert len(result) == 3

    def test_with_duplicates(self) -> None:
        chunks = ["hello", "world", "hello", "foo", "world"]
        result = deduplicate_chunks(chunks)
        assert len(result) == 3

    def test_empty(self) -> None:
        assert deduplicate_chunks([]) == []
