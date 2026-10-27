"""Tests for scripts/k8s_ingest_ollama.py.

All external services (ChromaDB, Neo4j, Ollama) are mocked.
These tests verify the orchestration logic, not external calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from spiral_02_graph.entity_extractor import ExtractionResult


class TestIngestViaOllama:
    """Tests for the ingest_via_ollama function."""

    @patch("scripts.k8s_ingest_ollama.GraphStore")
    @patch("scripts.k8s_ingest_ollama.OllamaEntityExtractor")
    @patch("scripts.k8s_ingest_ollama.VectorStore")
    def test_successful_ingest(self, mock_store_cls, mock_extractor_cls, mock_graph_cls):
        """All chunks processed successfully."""
        # Setup VectorStore mock
        mock_store = mock_store_cls.return_value
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["chunk_0", "chunk_1"],
            "documents": ["FastAPI is a framework", "Django is also a framework"],
        }
        mock_store._collection = mock_collection

        # Setup Extractor mock
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_from_chunk.return_value = ExtractionResult(
            entities=[MagicMock()],
            relations=[MagicMock()],
            chunk_id="chunk_0",
            model="mistral:7b",
        )

        # Setup GraphStore mock
        mock_graph = mock_graph_cls.return_value
        mock_graph.node_count = 0
        mock_graph.edge_count = 0

        from scripts.k8s_ingest_ollama import ingest_via_ollama

        summary = ingest_via_ollama(
            ollama_url="http://localhost:11434",
            model="mistral:7b",
        )

        assert summary["total_chunks"] == 2
        assert summary["success"] == 2
        assert summary["fail"] == 0
        assert summary["entities_extracted"] >= 2
        assert summary["relations_extracted"] >= 2
        assert summary["model"] == "mistral:7b"

    @patch("scripts.k8s_ingest_ollama.GraphStore")
    @patch("scripts.k8s_ingest_ollama.OllamaEntityExtractor")
    @patch("scripts.k8s_ingest_ollama.VectorStore")
    def test_partial_failure(self, mock_store_cls, mock_extractor_cls, mock_graph_cls):
        """Some chunks fail extraction but process continues."""
        mock_store = mock_store_cls.return_value
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["chunk_0", "chunk_1", "chunk_2"],
            "documents": ["text_0", "text_1", "text_2"],
        }
        mock_store._collection = mock_collection

        mock_extractor = mock_extractor_cls.return_value
        call_count = 0

        def side_effect(text, chunk_id):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Ollama connection refused")
            return ExtractionResult(
                entities=[], relations=[], chunk_id=chunk_id, model="mistral:7b"
            )

        mock_extractor.extract_from_chunk.side_effect = side_effect

        mock_graph = mock_graph_cls.return_value
        mock_graph.node_count = 0
        mock_graph.edge_count = 0

        from scripts.k8s_ingest_ollama import ingest_via_ollama

        summary = ingest_via_ollama()

        assert summary["total_chunks"] == 3
        assert summary["success"] == 2
        assert summary["fail"] == 1

    @patch("scripts.k8s_ingest_ollama.GraphStore")
    @patch("scripts.k8s_ingest_ollama.OllamaEntityExtractor")
    @patch("scripts.k8s_ingest_ollama.VectorStore")
    def test_empty_store(self, mock_store_cls, mock_extractor_cls, mock_graph_cls):
        """No chunks in ChromaDB returns zero counts."""
        mock_store = mock_store_cls.return_value
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "documents": []}
        mock_store._collection = mock_collection

        mock_graph = mock_graph_cls.return_value
        mock_graph.node_count = 5
        mock_graph.edge_count = 10

        from scripts.k8s_ingest_ollama import ingest_via_ollama

        summary = ingest_via_ollama()

        assert summary["total_chunks"] == 0
        assert summary["success"] == 0
        assert summary["fail"] == 0
        assert summary["nodes_before"] == 5

    @patch("scripts.k8s_ingest_ollama.GraphStore")
    @patch("scripts.k8s_ingest_ollama.OllamaEntityExtractor")
    @patch("scripts.k8s_ingest_ollama.VectorStore")
    def test_batch_progress_reporting(
        self, mock_store_cls, mock_extractor_cls, mock_graph_cls, capsys
    ):
        """Progress is reported at batch boundaries."""
        mock_store = mock_store_cls.return_value
        mock_collection = MagicMock()
        # Create 30 chunks to trigger batch reporting (batch_size=25)
        ids = [f"chunk_{i}" for i in range(30)]
        docs = [f"text_{i}" for i in range(30)]
        mock_collection.get.return_value = {"ids": ids, "documents": docs}
        mock_store._collection = mock_collection

        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_from_chunk.return_value = ExtractionResult(
            entities=[], relations=[], chunk_id="x", model="mistral:7b"
        )

        mock_graph = mock_graph_cls.return_value
        mock_graph.node_count = 0
        mock_graph.edge_count = 0

        from scripts.k8s_ingest_ollama import ingest_via_ollama

        summary = ingest_via_ollama(batch_size=25)

        captured = capsys.readouterr()
        assert "[25/30]" in captured.out
        assert summary["total_chunks"] == 30


class TestMainCLI:
    """Tests for the CLI entry point."""

    @patch("sys.argv", ["k8s_ingest_ollama.py"])
    @patch("scripts.k8s_ingest_ollama.ingest_via_ollama")
    def test_main_exits_zero_on_success(self, mock_ingest):
        """main() exits 0 when no failures."""
        mock_ingest.return_value = {"fail": 0}

        from scripts.k8s_ingest_ollama import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("sys.argv", ["k8s_ingest_ollama.py"])
    @patch("scripts.k8s_ingest_ollama.ingest_via_ollama")
    def test_main_exits_one_on_failure(self, mock_ingest):
        """main() exits 1 when there are failures."""
        mock_ingest.return_value = {"fail": 3}

        from scripts.k8s_ingest_ollama import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
