"""Tests for QueryExpander and HyDEExpander."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spiral_01_chunks.query_expander import HyDEExpander, QueryExpander


class TestQueryExpander:
    def test_original_always_first(self) -> None:
        expander = QueryExpander()
        variations = expander.expand("How do I create a route in FastAPI?")
        assert variations[0] == "How do I create a route in FastAPI?"

    def test_returns_list(self) -> None:
        expander = QueryExpander()
        result = expander.expand("What is dependency injection?")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_max_variations_respected(self) -> None:
        expander = QueryExpander(max_variations=2)
        result = expander.expand("How do I configure middleware in FastAPI?")
        assert len(result) <= 2

    def test_keyword_variation_removes_stop_words(self) -> None:
        expander = QueryExpander(max_variations=3)
        variations = expander.expand("What is the best way to handle errors?")
        keywords_var = variations[1] if len(variations) > 1 else ""
        # Stop words like "what", "is", "the", "to" should not appear
        for stop in ["what", "is", "the"]:
            assert stop not in keywords_var.lower().split()

    def test_question_to_statement_removes_how_do_i(self) -> None:
        expander = QueryExpander(max_variations=3)
        variations = expander.expand("How do I add middleware?")
        statements = [v for v in variations if "how do i" not in v.lower()]
        assert len(statements) >= 1

    def test_no_duplicate_variations(self) -> None:
        expander = QueryExpander(max_variations=5)
        variations = expander.expand("FastAPI middleware")
        assert len(variations) == len(set(variations))

    def test_empty_query_returns_list(self) -> None:
        expander = QueryExpander()
        result = expander.expand("")
        assert isinstance(result, list)

    def test_short_query_just_returns_original(self) -> None:
        expander = QueryExpander()
        result = expander.expand("FastAPI")
        assert "FastAPI" in result


class TestHyDEExpander:
    def test_raises_without_api_key(self) -> None:
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            HyDEExpander(api_key="")

    def test_expand_returns_two_elements(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "FastAPI uses path decorators to define routes."

        with patch("spiral_01_chunks.query_expander.Groq") as mock_groq_cls:
            mock_client = mock_groq_cls.return_value
            mock_client.chat.completions.create.return_value = mock_response

            expander = HyDEExpander(api_key="fake-key")
            result = expander.expand("How to create a route in FastAPI?")

        assert len(result) == 2

    def test_original_query_is_first(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "A hypothetical answer."

        with patch("spiral_01_chunks.query_expander.Groq") as mock_groq_cls:
            mock_client = mock_groq_cls.return_value
            mock_client.chat.completions.create.return_value = mock_response

            expander = HyDEExpander(api_key="fake-key")
            result = expander.expand("What is FastAPI?")

        assert result[0] == "What is FastAPI?"

    def test_hyde_document_is_llm_output(self) -> None:
        hypothetical = "FastAPI is a modern Python web framework based on type hints."
        mock_response = MagicMock()
        mock_response.choices[0].message.content = hypothetical

        with patch("spiral_01_chunks.query_expander.Groq") as mock_groq_cls:
            mock_client = mock_groq_cls.return_value
            mock_client.chat.completions.create.return_value = mock_response

            expander = HyDEExpander(api_key="fake-key")
            result = expander.expand("What is FastAPI?")

        assert result[1] == hypothetical

    def test_fallback_when_llm_returns_empty(self) -> None:
        """When LLM returns empty string, fallback to original query."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""

        with patch("spiral_01_chunks.query_expander.Groq") as mock_groq_cls:
            mock_client = mock_groq_cls.return_value
            mock_client.chat.completions.create.return_value = mock_response

            expander = HyDEExpander(api_key="fake-key")
            result = expander.expand("What is FastAPI?")

        assert result[1] == "What is FastAPI?"
