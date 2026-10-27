"""Tests for Spiral 01 — Pipeline module.

All external dependencies (VectorStore, Retriever, Generator) are mocked.
Tests validate:
- RAGPipeline orchestration (ingest, ask, stats, compare_strategies)
- Lazy generator loading
- CLI entry point (main)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spiral_01_chunks.chunker import Chunk
from spiral_01_chunks.retriever import RetrievalResult

# --- Fixtures ---


@pytest.fixture
def docs_dir():
    """Create a temp dir with a sample .md file for ingestion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "test.md").write_text(
            "FastAPI is a modern web framework. It uses Python type hints for validation.",
            encoding="utf-8",
        )
        yield str(p)


def _make_retrieval_result(text="FastAPI is modern", source="test.md", similarity=0.9):
    return RetrievalResult(
        text=text, source=source, chunk_index=0, similarity=similarity, metadata={}
    )


@pytest.fixture
def mock_store():
    """Mock VectorStore."""
    store = MagicMock()
    store.add_documents.return_value = 3
    store.count = 3
    store.stats.return_value = {
        "collection": "test",
        "total_chunks": 3,
        "embedding_model": "all-MiniLM-L6-v2",
        "similarity_metric": "cosine",
    }
    store.query.return_value = [
        {
            "text": "FastAPI is modern",
            "metadata": {"source": "test.md", "chunk_index": 0},
            "similarity": 0.9,
            "id": "1",
        },
    ]
    store.clear.return_value = None
    return store


def _make_mock_gen_result(answer="FastAPI is a web framework."):
    gen_result = MagicMock()
    gen_result.answer = answer
    gen_result.model = "llama-3.3-70b-versatile"
    gen_result.total_tokens = 100
    gen_result.estimated_cost = 0.0
    return gen_result


@pytest.fixture
def mock_generator():
    """Mock Generator."""
    gen = MagicMock()
    gen.generate.return_value = _make_mock_gen_result()
    return gen


@pytest.fixture
def pipeline(mock_store, mock_generator):
    """Create a RAGPipeline with mocked internals."""
    with patch("spiral_01_chunks.pipeline.VectorStore", return_value=mock_store):
        with patch("spiral_01_chunks.pipeline.Retriever") as MockRetriever:
            retriever_instance = MagicMock()
            retriever_instance.retrieve.return_value = [_make_retrieval_result()]
            retriever_instance.retrieve_with_context.return_value = {
                "query": "test",
                "top_k": 5,
                "threshold": 0.3,
                "total_results": 1,
                "results": [_make_retrieval_result()],
                "avg_similarity": 0.9,
                "store_stats": mock_store.stats(),
            }
            MockRetriever.return_value = retriever_instance

            from spiral_01_chunks.pipeline import RAGPipeline

            p = RAGPipeline()
            p._generator = mock_generator
            yield p


# --- RAGPipeline.ingest ---


class TestPipelineIngest:
    def test_ingest_returns_stats(self, pipeline, docs_dir):
        with patch("spiral_01_chunks.pipeline.chunk_directory") as mock_cd:
            mock_cd.return_value = [
                Chunk(text="chunk1", source="test.md", chunk_index=0, strategy="recursive"),
                Chunk(text="chunk2", source="test.md", chunk_index=1, strategy="recursive"),
            ]
            result = pipeline.ingest(docs_dir)
            assert "chunks_created" in result
            assert "strategy" in result
            assert result["chunks_created"] == 2

    def test_ingest_custom_strategy(self, pipeline, docs_dir):
        with patch("spiral_01_chunks.pipeline.chunk_directory") as mock_cd:
            mock_cd.return_value = []
            pipeline.ingest(docs_dir, strategy="sentence")
            mock_cd.assert_called_once()
            assert "sentence" in str(mock_cd.call_args)


# --- RAGPipeline.ask ---


class TestPipelineAsk:
    def test_ask_returns_answer(self, pipeline):
        result = pipeline.ask("What is FastAPI?")
        assert "answer" in result
        assert result["answer"] == "FastAPI is a web framework."

    def test_ask_returns_sources(self, pipeline):
        result = pipeline.ask("What is FastAPI?")
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_ask_returns_timing(self, pipeline):
        result = pipeline.ask("What is FastAPI?")
        assert "retrieval_time_s" in result
        assert "generation_time_s" in result

    def test_ask_includes_model_and_cost(self, pipeline):
        result = pipeline.ask("What is FastAPI?")
        assert "model" in result
        assert "cost" in result
        assert "tokens" in result

    def test_ask_show_sources_false(self, pipeline):
        result = pipeline.ask("What is FastAPI?", show_sources=False)
        assert "answer" in result

    def test_ask_no_results(self, pipeline):
        pipeline._retriever.retrieve_with_context.return_value = {
            "query": "test",
            "top_k": 5,
            "threshold": 0.3,
            "total_results": 0,
            "results": [],
            "avg_similarity": 0.0,
            "store_stats": {},
        }
        result = pipeline.ask("Unknown topic")
        assert result["answer"] == "No relevant context found."


# --- RAGPipeline.stats ---


class TestPipelineStats:
    def test_stats_returns_dict(self, pipeline):
        result = pipeline.stats()
        assert "total_chunks" in result
        assert result["total_chunks"] == 3


# --- RAGPipeline.compare_strategies ---


class TestPipelineCompare:
    def test_compare_runs_all_strategies(self, mock_generator):
        with patch("spiral_01_chunks.pipeline.VectorStore") as MockVS:
            mock_s = MagicMock()
            mock_s.count = 2
            mock_s.add_documents.return_value = 2
            mock_s.clear.return_value = None
            MockVS.return_value = mock_s

            with patch("spiral_01_chunks.pipeline.Retriever") as MockRet:
                ret_inst = MagicMock()
                ret_inst.retrieve.return_value = [_make_retrieval_result()]
                MockRet.return_value = ret_inst

                with patch("spiral_01_chunks.pipeline.chunk_directory") as mock_cd:
                    mock_cd.return_value = [
                        Chunk(text="c1", source="t.md", chunk_index=0, strategy="fixed"),
                    ]

                    from spiral_01_chunks.pipeline import RAGPipeline

                    p = RAGPipeline()
                    p._generator = mock_generator
                    p.compare_strategies("What is FastAPI?", strategies=["fixed", "recursive"])

                    assert mock_cd.call_count == 2
                    assert mock_s.clear.call_count >= 2

    def test_compare_handles_generator_failure(self, mock_generator):
        mock_generator.generate.side_effect = Exception("No API key")

        with patch("spiral_01_chunks.pipeline.VectorStore") as MockVS:
            mock_s = MagicMock()
            mock_s.count = 1
            mock_s.add_documents.return_value = 1
            mock_s.clear.return_value = None
            MockVS.return_value = mock_s

            with patch("spiral_01_chunks.pipeline.Retriever") as MockRet:
                ret_inst = MagicMock()
                ret_inst.retrieve.return_value = [_make_retrieval_result()]
                MockRet.return_value = ret_inst

                with patch("spiral_01_chunks.pipeline.chunk_directory") as mock_cd:
                    mock_cd.return_value = [
                        Chunk(text="c1", source="t.md", chunk_index=0, strategy="fixed"),
                    ]

                    from spiral_01_chunks.pipeline import RAGPipeline

                    p = RAGPipeline()
                    p._generator = mock_generator
                    # Should not raise — catches exception and shows fallback
                    p.compare_strategies("test?", strategies=["fixed"])

    def test_compare_no_results(self, mock_generator):
        with patch("spiral_01_chunks.pipeline.VectorStore") as MockVS:
            mock_s = MagicMock()
            mock_s.count = 1
            mock_s.add_documents.return_value = 1
            mock_s.clear.return_value = None
            MockVS.return_value = mock_s

            with patch("spiral_01_chunks.pipeline.Retriever") as MockRet:
                ret_inst = MagicMock()
                ret_inst.retrieve.return_value = []  # no results
                MockRet.return_value = ret_inst

                with patch("spiral_01_chunks.pipeline.chunk_directory") as mock_cd:
                    mock_cd.return_value = [
                        Chunk(text="c1", source="t.md", chunk_index=0, strategy="fixed"),
                    ]

                    from spiral_01_chunks.pipeline import RAGPipeline

                    p = RAGPipeline()
                    p._generator = mock_generator
                    p.compare_strategies("test?", strategies=["fixed"])


# --- RAGPipeline lazy generator ---


class TestPipelineLazyGenerator:
    def test_generator_is_lazy(self):
        with patch("spiral_01_chunks.pipeline.VectorStore"):
            with patch("spiral_01_chunks.pipeline.Retriever"):
                from spiral_01_chunks.pipeline import RAGPipeline

                p = RAGPipeline()
                assert p._generator is None

    def test_generator_property_initializes(self):
        with patch("spiral_01_chunks.pipeline.VectorStore"):
            with patch("spiral_01_chunks.pipeline.Retriever"):
                with patch("spiral_01_chunks.pipeline.Generator") as MockGen:
                    MockGen.return_value = MagicMock()
                    from spiral_01_chunks.pipeline import RAGPipeline

                    p = RAGPipeline()
                    gen = p.generator
                    assert gen is not None
                    MockGen.assert_called_once()


# --- CLI main ---


class TestPipelineMain:
    def test_no_args_exits(self):
        with patch("sys.argv", ["pipeline"]):
            with pytest.raises(SystemExit) as exc_info:
                from spiral_01_chunks.pipeline import main

                main()
            assert exc_info.value.code == 1

    def test_unknown_command_exits(self):
        with patch("sys.argv", ["pipeline", "foobar"]):
            with pytest.raises(SystemExit) as exc_info:
                from spiral_01_chunks.pipeline import main

                main()
            assert exc_info.value.code == 1

    def test_ingest_missing_path_exits(self):
        with patch("sys.argv", ["pipeline", "ingest"]):
            with pytest.raises(SystemExit) as exc_info:
                from spiral_01_chunks.pipeline import main

                main()
            assert exc_info.value.code == 1

    def test_ask_missing_question_exits(self):
        with patch("sys.argv", ["pipeline", "ask"]):
            with pytest.raises(SystemExit) as exc_info:
                from spiral_01_chunks.pipeline import main

                main()
            assert exc_info.value.code == 1

    def test_compare_missing_question_exits(self):
        with patch("sys.argv", ["pipeline", "compare"]):
            with pytest.raises(SystemExit) as exc_info:
                from spiral_01_chunks.pipeline import main

                main()
            assert exc_info.value.code == 1

    def test_stats_command_runs(self):
        with patch("sys.argv", ["pipeline", "stats"]):
            with patch("spiral_01_chunks.pipeline.RAGPipeline") as MockPipeline:
                mock_instance = MagicMock()
                mock_instance.stats.return_value = {"total_chunks": 0}
                MockPipeline.return_value = mock_instance
                from spiral_01_chunks.pipeline import main

                main()
                mock_instance.stats.assert_called_once()

    def test_ingest_command_runs(self, docs_dir):
        with patch("sys.argv", ["pipeline", "ingest", docs_dir]):
            with patch("spiral_01_chunks.pipeline.RAGPipeline") as MockPipeline:
                mock_instance = MagicMock()
                mock_instance.ingest.return_value = {"chunks_created": 2}
                MockPipeline.return_value = mock_instance
                from spiral_01_chunks.pipeline import main

                main()
                mock_instance.ingest.assert_called_once_with(docs_dir)

    def test_ask_command_runs(self):
        with patch("sys.argv", ["pipeline", "ask", "What", "is", "FastAPI?"]):
            with patch("spiral_01_chunks.pipeline.RAGPipeline") as MockPipeline:
                mock_instance = MagicMock()
                mock_instance.ask.return_value = {"answer": "test"}
                MockPipeline.return_value = mock_instance
                from spiral_01_chunks.pipeline import main

                main()
                mock_instance.ask.assert_called_once_with("What is FastAPI?")

    def test_compare_command_runs(self):
        with patch("sys.argv", ["pipeline", "compare", "test", "question"]):
            with patch("spiral_01_chunks.pipeline.RAGPipeline") as MockPipeline:
                mock_instance = MagicMock()
                MockPipeline.return_value = mock_instance
                from spiral_01_chunks.pipeline import main

                main()
                mock_instance.compare_strategies.assert_called_once_with("test question")
