"""Tests for Spiral 01 — Generator module.

All Groq API calls are mocked. Tests validate:
- Generator initialization and API key validation
- Prompt templating (SYSTEM_PROMPT, USER_TEMPLATE)
- Response parsing into GenerationResult
- generate() and generate_without_context() flows
- GenerationResult dataclass and estimated_cost property
"""

from unittest.mock import MagicMock, patch

import pytest

from spiral_01_chunks.generator import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    GenerationResult,
    Generator,
)

# --- Fixtures ---


def _make_mock_response(
    content="Mocked answer", prompt_tokens=50, completion_tokens=30, total_tokens=80
):
    """Build a mock Groq chat completion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.fixture
def mock_groq():
    """Patch Groq client so no real API calls are made."""
    with patch("spiral_01_chunks.generator.Groq") as MockGroq:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response()
        MockGroq.return_value = mock_client
        yield mock_client


@pytest.fixture
def generator(mock_groq):
    """Create a Generator with mocked Groq client."""
    return Generator(api_key="gsk_fake_test_key_1234567890")


# --- GenerationResult ---


class TestGenerationResult:
    def test_has_required_fields(self):
        result = GenerationResult(
            answer="Hello",
            model="llama-3.3-70b",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        assert result.answer == "Hello"
        assert result.model == "llama-3.3-70b"
        assert result.total_tokens == 15

    def test_estimated_cost_is_zero_for_groq(self):
        result = GenerationResult(
            answer="Test",
            model="llama-3.3-70b",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        assert result.estimated_cost == 0.0


# --- Generator init ---


class TestGeneratorInit:
    def test_raises_without_api_key(self):
        with pytest.raises(ValueError, match="GROQ_API_KEY not set"):
            Generator(api_key="")

    def test_creates_with_valid_key(self, mock_groq):
        gen = Generator(api_key="gsk_valid_key")
        assert gen._model == "llama-3.3-70b-versatile"

    def test_accepts_custom_params(self, mock_groq):
        gen = Generator(
            api_key="gsk_valid_key",
            model="custom-model",
            temperature=0.5,
            max_tokens=512,
        )
        assert gen._model == "custom-model"
        assert gen._temperature == 0.5
        assert gen._max_tokens == 512


# --- Generator.generate ---


class TestGeneratorGenerate:
    def test_returns_generation_result(self, generator, mock_groq):
        result = generator.generate("some context", "what is FastAPI?")
        assert isinstance(result, GenerationResult)
        assert result.answer == "Mocked answer"

    def test_calls_groq_with_correct_messages(self, generator, mock_groq):
        generator.generate("my context", "my question")
        call_args = mock_groq.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "my context" in messages[1]["content"]
        assert "my question" in messages[1]["content"]

    def test_uses_user_template(self, generator, mock_groq):
        generator.generate("ctx_data", "q_data")
        call_args = mock_groq.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        expected = USER_TEMPLATE.format(context="ctx_data", question="q_data")
        assert user_msg == expected

    def test_passes_model_and_temperature(self, generator, mock_groq):
        generator.generate("ctx", "q")
        call_args = mock_groq.chat.completions.create.call_args
        assert call_args.kwargs["model"] == generator._model
        assert call_args.kwargs["temperature"] == generator._temperature
        assert call_args.kwargs["max_tokens"] == generator._max_tokens

    def test_handles_none_content(self, generator, mock_groq):
        mock_groq.chat.completions.create.return_value = _make_mock_response(content=None)
        result = generator.generate("ctx", "q")
        assert result.answer == ""

    def test_handles_none_usage(self, generator, mock_groq):
        resp = _make_mock_response()
        resp.usage = None
        mock_groq.chat.completions.create.return_value = resp
        result = generator.generate("ctx", "q")
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    def test_token_counts_propagated(self, generator, mock_groq):
        mock_groq.chat.completions.create.return_value = _make_mock_response(
            prompt_tokens=100,
            completion_tokens=42,
            total_tokens=142,
        )
        result = generator.generate("ctx", "q")
        assert result.input_tokens == 100
        assert result.output_tokens == 42
        assert result.total_tokens == 142


# --- Generator.generate_without_context ---


class TestGeneratorWithoutContext:
    def test_returns_generation_result(self, generator, mock_groq):
        result = generator.generate_without_context("what is Python?")
        assert isinstance(result, GenerationResult)

    def test_does_not_use_system_prompt(self, generator, mock_groq):
        generator.generate_without_context("test question")
        call_args = mock_groq.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] != SYSTEM_PROMPT  # uses simpler prompt

    def test_passes_question_directly(self, generator, mock_groq):
        generator.generate_without_context("direct question here")
        call_args = mock_groq.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert user_msg == "direct question here"


# --- Key rotation / rate-limit retry ---


class TestGeneratorKeyRotation:
    """Tests for _call_with_rotation rate-limit retry and fallback."""

    def test_retries_on_429_and_raises_last_error(self):
        """All keys return 429 → raises the last error."""
        with patch("spiral_01_chunks.generator.Groq") as MockGroq:
            client1 = MagicMock()
            client1.chat.completions.create.side_effect = Exception("429 rate_limit")
            MockGroq.return_value = client1
            gen = Generator(api_key="key1")

            with pytest.raises(Exception, match="429"):
                gen.generate("ctx", "q")

    def test_non_rate_limit_error_raises_immediately(self):
        """Non-429 errors are raised immediately, not retried."""
        with patch("spiral_01_chunks.generator.Groq") as MockGroq:
            client = MagicMock()
            client.chat.completions.create.side_effect = ValueError("bad input")
            MockGroq.return_value = client
            gen = Generator(api_key="key1")

            with pytest.raises(ValueError, match="bad input"):
                gen.generate("ctx", "q")
