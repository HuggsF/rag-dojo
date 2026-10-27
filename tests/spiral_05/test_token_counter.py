"""Tests for spiral_05_economics.token_counter — token counting."""

from spiral_05_economics.token_counter import TokenCount


class TestTokenCount:
    """Tests for TokenCount dataclass."""

    def test_has_required_fields(self):
        tc = TokenCount(input_tokens=100, output_tokens=50, model="test")
        assert tc.input_tokens == 100
        assert tc.output_tokens == 50
        assert tc.model == "test"

    def test_total_tokens(self):
        tc = TokenCount(input_tokens=100, output_tokens=50, model="test")
        assert tc.total == 150

    def test_repr(self):
        tc = TokenCount(input_tokens=100, output_tokens=50, model="test")
        assert "100" in repr(tc)
        assert "50" in repr(tc)


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_text_returns_positive(self, token_counter):
        count = token_counter.count_text("Hello world, this is a test.")
        assert count > 0

    def test_empty_text_returns_zero(self, token_counter):
        count = token_counter.count_text("")
        assert count == 0

    def test_longer_text_has_more_tokens(self, token_counter):
        short = token_counter.count_text("Hello")
        long = token_counter.count_text("Hello world this is a much longer text with many words")
        assert long > short

    def test_count_prompt_and_response(self, token_counter):
        result = token_counter.count_prompt_and_response(
            prompt="What is FastAPI?",
            response="FastAPI is a modern Python web framework.",
            model="llama-3.3-70b-versatile",
        )
        assert isinstance(result, TokenCount)
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.model == "llama-3.3-70b-versatile"

    def test_estimate_prompt_tokens(self, token_counter):
        """Estimate tokens before sending (no response yet)."""
        result = token_counter.estimate_prompt(
            prompt="Explain middleware in FastAPI",
            model="llama-3.3-70b-versatile",
        )
        assert isinstance(result, TokenCount)
        assert result.input_tokens > 0
        assert result.output_tokens == 0

    def test_fallback_for_unknown_encoding(self, token_counter):
        """Should fall back to char-based estimation for unknown models."""
        count = token_counter.count_text("Hello world", model="unknown-model-xyz")
        assert count > 0

    def test_get_encoder_returns_none_on_exception(self, token_counter):
        """Cover _get_encoder returning None when tiktoken raises (lines 61-62)."""
        from unittest.mock import patch

        with patch(
            "spiral_05_economics.token_counter.tiktoken.get_encoding",
            side_effect=Exception("broken"),
        ):
            # Force fresh lookup by using a model not yet cached
            encoder = token_counter._get_encoder("never-seen-model")
            assert encoder is None

    def test_count_text_char_fallback_when_encoder_none(self, token_counter):
        """Cover char/4 fallback path when encoder is None (line 74)."""
        from unittest.mock import patch

        with patch.object(token_counter, "_get_encoder", return_value=None):
            count = token_counter.count_text(
                "Hello world, this is a test string.", model="any-model"
            )
            # char/4 estimation: max(1, len("Hello world, this is a test string.") // 4)
            assert count == max(1, len("Hello world, this is a test string.") // 4)
