"""Tests for Spiral 01 — OllamaGenerator module.

All HTTP calls are mocked. Tests validate:
- OllamaGenerator initialization with custom params
- generate() calls Ollama API with correct payload
- Response parsing into GenerationResult
- Error handling for HTTP failures
"""

from unittest.mock import MagicMock, patch

import pytest

from spiral_01_chunks.generator import GenerationResult
from spiral_01_chunks.ollama_generator import OllamaGenerator


class TestOllamaGeneratorInit:
    def test_default_params(self):
        gen = OllamaGenerator()
        assert gen._model == "qwen2.5:3b"
        assert "11434" in gen._base_url
        assert gen._temperature == 0.1
        assert gen._max_tokens == 1024

    def test_custom_params(self):
        gen = OllamaGenerator(
            base_url="http://localhost:11434/",
            model="mistral:7b",
            temperature=0.5,
            max_tokens=512,
        )
        assert gen._model == "mistral:7b"
        assert gen._base_url == "http://localhost:11434"  # trailing slash stripped
        assert gen._temperature == 0.5
        assert gen._max_tokens == 512


class TestOllamaGeneratorGenerate:
    @patch("spiral_01_chunks.ollama_generator.httpx.post")
    def test_returns_generation_result(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Ollama answer"},
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        gen = OllamaGenerator(base_url="http://localhost:11434")
        result = gen.generate("some context", "what is FastAPI?")

        assert isinstance(result, GenerationResult)
        assert result.answer == "Ollama answer"
        assert result.model == "qwen2.5:3b"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    @patch("spiral_01_chunks.ollama_generator.httpx.post")
    def test_calls_correct_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        gen = OllamaGenerator(base_url="http://localhost:11434")
        gen.generate("ctx", "q")

        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["timeout"] == 300.0

    @patch("spiral_01_chunks.ollama_generator.httpx.post")
    def test_sends_correct_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        gen = OllamaGenerator(model="mistral:7b", temperature=0.3, max_tokens=256)
        gen.generate("my context", "my question")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "mistral:7b"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.3
        assert payload["options"]["num_predict"] == 256

    @patch("spiral_01_chunks.ollama_generator.httpx.post")
    def test_handles_missing_fields_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}  # empty response
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        gen = OllamaGenerator()
        result = gen.generate("ctx", "q")

        assert result.answer == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    @patch("spiral_01_chunks.ollama_generator.httpx.post")
    def test_raises_on_http_error(self, mock_post):
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_resp

        gen = OllamaGenerator()
        with pytest.raises(httpx.HTTPStatusError):
            gen.generate("ctx", "q")
