"""Tests for Spiral 02 — Entity Extractor module.

Key validations:
- Entity and Relation dataclasses
- JSON parsing (clean, code-fenced, malformed)
- Label/relation type validation with fallbacks
- Extraction prompt format
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from spiral_02_graph.entity_extractor import (
    Entity,
    EntityExtractor,
    ExtractionResult,
    OllamaEntityExtractor,
    Relation,
)


class TestEntity:
    def test_basic_creation(self):
        e = Entity(name="FastAPI", label="Module")
        assert e.name == "FastAPI"
        assert e.label == "Module"
        assert e.properties == {}

    def test_name_stripped(self):
        e = Entity(name="  FastAPI  ", label="Module")
        assert e.name == "FastAPI"

    def test_invalid_label_fallback(self):
        e = Entity(name="Something", label="InvalidLabel")
        assert e.label == "Concept"  # fallback

    def test_with_properties(self):
        e = Entity(name="DI", label="Concept", properties={"description": "Dependency Injection"})
        assert e.properties["description"] == "Dependency Injection"

    def test_repr(self):
        e = Entity(name="FastAPI", label="Module")
        assert "Module" in repr(e)
        assert "FastAPI" in repr(e)


class TestRelation:
    def test_basic_creation(self):
        r = Relation(source="FastAPI", target="Starlette", relation_type="USES")
        assert r.source == "FastAPI"
        assert r.target == "Starlette"
        assert r.relation_type == "USES"

    def test_names_stripped(self):
        r = Relation(source="  FastAPI  ", target="  Starlette  ", relation_type="USES")
        assert r.source == "FastAPI"
        assert r.target == "Starlette"

    def test_invalid_relation_fallback(self):
        r = Relation(source="A", target="B", relation_type="UNKNOWN_TYPE")
        assert r.relation_type == "RELATED_TO"  # fallback

    def test_repr(self):
        r = Relation(source="FastAPI", target="Starlette", relation_type="USES")
        assert "USES" in repr(r)


class TestExtractionResult:
    def test_total_tokens(self, sample_extraction_result: ExtractionResult):
        assert sample_extraction_result.total_tokens == 700  # 500 + 200

    def test_empty_result(self):
        r = ExtractionResult(entities=[], relations=[], chunk_id="test", model="test")
        assert r.total_tokens == 0


class TestParseResponse:
    """Test the JSON parser for LLM responses."""

    def test_clean_json(self):
        response = json.dumps(
            {
                "entities": [
                    {"name": "FastAPI", "label": "Module", "description": "Web framework"},
                ],
                "relations": [
                    {"source": "FastAPI", "target": "FastAPI", "type": "RELATED_TO"},
                ],
            }
        )
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.entities) == 1
        assert result.entities[0].name == "FastAPI"

    def test_json_in_code_fence(self):
        response = (
            '```json\n{"entities": [{"name": "Test", "label": "Concept"}], "relations": []}\n```'
        )
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.entities) == 1
        assert result.entities[0].name == "Test"

    def test_malformed_json_returns_empty(self):
        result = EntityExtractor._parse_llm_response("not json at all", "test_chunk")
        assert len(result.entities) == 0
        assert len(result.relations) == 0

    def test_empty_response(self):
        result = EntityExtractor._parse_llm_response("", "test_chunk")
        assert len(result.entities) == 0

    def test_missing_entities_key(self):
        response = json.dumps({"relations": []})
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.entities) == 0

    def test_relations_only_between_known_entities(self):
        response = json.dumps(
            {
                "entities": [
                    {"name": "A", "label": "Concept"},
                    {"name": "B", "label": "Concept"},
                ],
                "relations": [
                    {"source": "A", "target": "B", "type": "USES"},
                    {"source": "A", "target": "C", "type": "USES"},  # C not in entities
                ],
            }
        )
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.relations) == 1  # only A->B

    def test_max_entities_limit(self):
        entities = [{"name": f"Entity_{i}", "label": "Concept"} for i in range(20)]
        response = json.dumps({"entities": entities, "relations": []})
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.entities) <= 10  # MAX_ENTITIES_PER_CHUNK

    def test_entity_with_invalid_label_gets_fallback(self):
        response = json.dumps(
            {
                "entities": [{"name": "X", "label": "INVALID_LABEL"}],
                "relations": [],
            }
        )
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert result.entities[0].label == "Concept"

    def test_json_with_surrounding_text(self):
        response = 'Here is the result:\n{"entities": [{"name": "Test", "label": "Module"}], "relations": []}\nDone!'
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.entities) == 1

    def test_relation_without_dict_skipped(self):
        response = json.dumps(
            {
                "entities": [{"name": "A", "label": "Concept"}],
                "relations": ["not a dict", {"source": "A", "target": "A", "type": "USES"}],
            }
        )
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.relations) == 1

    def test_entity_without_name_skipped(self):
        response = json.dumps(
            {
                "entities": [{"label": "Concept"}, {"name": "Valid", "label": "Module"}],
                "relations": [],
            }
        )
        result = EntityExtractor._parse_llm_response(response, "test_chunk")
        assert len(result.entities) == 1
        assert result.entities[0].name == "Valid"


class TestEntityExtractorInit:
    def test_raises_without_api_key(self):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            EntityExtractor(api_key="")

    def test_constructs_with_api_key(self):
        with patch("spiral_02_graph.entity_extractor.Groq") as mock_groq:
            extractor = EntityExtractor(api_key="test-key", model="test-model")
            mock_groq.assert_called_once_with(api_key="test-key")
            assert extractor._model == "test-model"


class TestExtractFromChunk:
    @pytest.fixture
    def extractor(self):
        with patch("spiral_02_graph.entity_extractor.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 50

            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps(
                {
                    "entities": [{"name": "FastAPI", "label": "Module"}],
                    "relations": [],
                }
            )

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_client.chat.completions.create.return_value = mock_response

            yield EntityExtractor(api_key="test-key")

    def test_returns_extraction_result(self, extractor):
        result = extractor.extract_from_chunk("Some text about FastAPI", "chunk_0")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 1
        assert result.entities[0].name == "FastAPI"

    def test_sets_model_and_tokens(self, extractor):
        result = extractor.extract_from_chunk("text", "chunk_0")
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_handles_none_usage(self):
        with patch("spiral_02_graph.entity_extractor.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client

            mock_choice = MagicMock()
            mock_choice.message.content = '{"entities": [], "relations": []}'

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_client.chat.completions.create.return_value = mock_response

            extractor = EntityExtractor(api_key="test-key")
            result = extractor.extract_from_chunk("text", "chunk_0")
            assert result.input_tokens == 0
            assert result.output_tokens == 0


class TestExtractFromChunks:
    @pytest.fixture
    def extractor(self):
        with patch("spiral_02_graph.entity_extractor.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 50
            mock_usage.completion_tokens = 30

            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps(
                {"entities": [{"name": "Test", "label": "Concept"}], "relations": []}
            )

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_client.chat.completions.create.return_value = mock_response

            yield EntityExtractor(api_key="test-key")

    def _make_chunk(self, text="some text", source="test.md", chunk_index=0):
        chunk = MagicMock()
        chunk.text = text
        chunk.source = source
        chunk.chunk_index = chunk_index
        return chunk

    def test_extracts_from_multiple_chunks(self, extractor):
        chunks = [self._make_chunk(chunk_index=i) for i in range(3)]
        results = extractor.extract_from_chunks(chunks, batch_delay=0)
        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_handles_extraction_error(self, extractor):
        client = extractor._clients[0]
        client.chat.completions.create.side_effect = [
            Exception("API error"),
            client.chat.completions.create.return_value,
        ]
        chunks = [self._make_chunk(chunk_index=0), self._make_chunk(chunk_index=1)]
        results = extractor.extract_from_chunks(chunks, batch_delay=0)
        assert len(results) == 2
        assert results[0].entities == []  # error chunk
        assert len(results[1].entities) == 1  # success chunk

    def test_rate_limiting_delay(self, extractor):
        chunks = [self._make_chunk(chunk_index=i) for i in range(2)]
        start = time.time()
        extractor.extract_from_chunks(chunks, batch_delay=0.05)
        elapsed = time.time() - start
        assert elapsed >= 0.04  # at least one delay applied


class TestOllamaEntityExtractor:
    def test_constructs_with_defaults(self):
        ext = OllamaEntityExtractor()
        assert ext._model == "mistral:7b"
        assert ext._base_url == "http://localhost:11434"

    def test_constructs_with_custom_params(self):
        ext = OllamaEntityExtractor(model="llama3.2:3b", base_url="http://myhost:9999")
        assert ext._model == "llama3.2:3b"
        assert ext._base_url == "http://myhost:9999"

    def test_extract_from_chunk(self):
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "content": json.dumps(
                        {
                            "entities": [{"name": "FastAPI", "label": "Module"}],
                            "relations": [],
                        }
                    )
                }
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            ext = OllamaEntityExtractor(model="mistral:7b")
            result = ext.extract_from_chunk("FastAPI text", "chunk_0")

            assert isinstance(result, ExtractionResult)
            assert len(result.entities) == 1
            assert result.entities[0].name == "FastAPI"
            assert result.model == "mistral:7b"

    def test_extract_from_chunks_with_error(self):
        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")

            ext = OllamaEntityExtractor(model="mistral:7b")
            chunk = MagicMock()
            chunk.text = "some text"
            chunk.source = "test.md"
            chunk.chunk_index = 0

            results = ext.extract_from_chunks([chunk], batch_delay=0)
            assert len(results) == 1
            assert results[0].entities == []
            assert results[0].model == "mistral:7b"

    def test_extract_from_chunks_success(self):
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "content": '{"entities": [{"name": "Test", "label": "Concept"}], "relations": []}'
                }
            }
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            ext = OllamaEntityExtractor(model="mistral:7b")
            chunk = MagicMock()
            chunk.text = "some text"
            chunk.source = "test.md"
            chunk.chunk_index = 0

            results = ext.extract_from_chunks([chunk], batch_delay=0)
            assert len(results) == 1
            assert len(results[0].entities) == 1
