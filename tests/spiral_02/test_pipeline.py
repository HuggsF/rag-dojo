"""Tests for spiral_02_graph.pipeline — GraphRAGPipeline orchestration."""

from unittest.mock import MagicMock, patch

import pytest

from spiral_02_graph.hybrid_retriever import HybridRetriever
from spiral_02_graph.pipeline import GraphRAGPipeline, main


class TestGraphRAGPipelineInit:
    """Tests for pipeline initialization."""

    def test_init_creates_vector_store(self):
        """Pipeline should create a vector store on init."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            assert pipeline._graph_store is None
            assert pipeline._extractor is None
            assert pipeline._hybrid_retriever is None
            assert pipeline._reranker is None
            assert pipeline._generator is None
            assert pipeline._visualizer is None


class TestLazyProperties:
    """Tests for lazy-loaded properties."""

    def test_graph_store_lazy_load(self):
        """graph_store property should lazy-load."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            with patch("spiral_02_graph.pipeline.GraphStore") as MockGS:
                mock_gs = MagicMock()
                MockGS.return_value = mock_gs
                result = pipeline.graph_store
                assert result is mock_gs

    def test_graph_store_cached(self):
        """Second access should return cached instance."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            mock_gs = MagicMock()
            pipeline._graph_store = mock_gs
            assert pipeline.graph_store is mock_gs

    def test_extractor_lazy_load(self):
        """extractor property should lazy-load."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            with patch("spiral_02_graph.pipeline.EntityExtractor") as MockEE:
                mock_ee = MagicMock()
                MockEE.return_value = mock_ee
                result = pipeline.extractor
                assert result is mock_ee

    def test_reranker_lazy_load(self):
        """reranker property should lazy-load."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            with patch("spiral_02_graph.pipeline.Reranker") as MockR:
                mock_r = MagicMock()
                MockR.return_value = mock_r
                result = pipeline.reranker
                assert result is mock_r

    def test_generator_lazy_load(self):
        """generator property should lazy-load."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            with patch("spiral_02_graph.pipeline.Generator") as MockG:
                mock_g = MagicMock()
                MockG.return_value = mock_g
                result = pipeline.generator
                assert result is mock_g

    def test_hybrid_retriever_lazy_load(self):
        """hybrid_retriever property should lazy-load."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            pipeline._graph_store = MagicMock()
            with patch("spiral_02_graph.pipeline.HybridRetriever") as MockHR:
                mock_hr = MagicMock()
                MockHR.return_value = mock_hr
                result = pipeline.hybrid_retriever
                assert result is mock_hr

    def test_visualizer_lazy_load(self):
        """visualizer property should lazy-load."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            pipeline._graph_store = MagicMock()
            with patch("spiral_02_graph.pipeline.GraphVisualizer") as MockV:
                mock_v = MagicMock()
                MockV.return_value = mock_v
                result = pipeline.visualizer
                assert result is mock_v


class TestIngestGraph:
    """Tests for ingest_graph method."""

    def test_ingest_no_chunks(self):
        """Should return error when no chunks in vector store."""
        with patch("spiral_02_graph.pipeline.VectorStore") as MockVS:
            mock_vs = MagicMock()
            mock_vs.stats.return_value = {"total_chunks": 0}
            MockVS.return_value = mock_vs
            pipeline = GraphRAGPipeline()
            result = pipeline.ingest_graph()
            assert result == {"error": "no chunks"}

    @patch("spiral_02_graph.pipeline.time.sleep")
    def test_ingest_with_chunks(self, mock_sleep):
        """Should process chunks and return stats."""
        with patch("spiral_02_graph.pipeline.VectorStore") as MockVS:
            mock_vs = MagicMock()
            mock_vs.stats.return_value = {"total_chunks": 2}
            MockVS.return_value = mock_vs

            pipeline = GraphRAGPipeline()

            # Mock retriever
            mock_result = MagicMock()
            mock_result.text = "test text"
            mock_result.source = "test.md"
            mock_result.chunk_index = 0
            with patch("spiral_02_graph.pipeline.Retriever") as MockR:
                MockR.return_value.retrieve.return_value = [mock_result]

                # Mock extractor
                mock_extraction = MagicMock()
                mock_extraction.entities = [MagicMock()]
                mock_extraction.relations = [MagicMock()]
                pipeline._extractor = MagicMock()
                pipeline._extractor.extract_from_chunk.return_value = mock_extraction

                # Mock graph store
                mock_gs = MagicMock()
                mock_gs.stats.return_value = {
                    "total_nodes": 1,
                    "total_edges": 1,
                    "labels": {"Concept": 1},
                    "relationship_types": {"USES": 1},
                    "dead_ends": 0,
                }
                pipeline._graph_store = mock_gs

                result = pipeline.ingest_graph()
                assert result["total_entities"] == 1
                assert result["total_relations"] == 1
                assert result["total_nodes"] == 1

    @patch("spiral_02_graph.pipeline.time.sleep")
    def test_ingest_handles_extraction_error(self, mock_sleep):
        """Should handle extraction errors gracefully."""
        with patch("spiral_02_graph.pipeline.VectorStore") as MockVS:
            mock_vs = MagicMock()
            mock_vs.stats.return_value = {"total_chunks": 1}
            MockVS.return_value = mock_vs

            pipeline = GraphRAGPipeline()

            mock_result = MagicMock()
            mock_result.text = "test"
            mock_result.source = "test.md"
            mock_result.chunk_index = 0
            with patch("spiral_02_graph.pipeline.Retriever") as MockR:
                MockR.return_value.retrieve.return_value = [mock_result]

                pipeline._extractor = MagicMock()
                pipeline._extractor.extract_from_chunk.side_effect = RuntimeError("API error")

                mock_gs = MagicMock()
                mock_gs.stats.return_value = {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "labels": {},
                    "relationship_types": {},
                    "dead_ends": 0,
                }
                pipeline._graph_store = mock_gs

                result = pipeline.ingest_graph()
                assert result["total_entities"] == 0


class TestAskHybrid:
    """Tests for ask_hybrid method."""

    def test_ask_no_results(self):
        """Should handle empty results."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            mock_hr = MagicMock()
            mock_hr.retrieve_with_context.return_value = {
                "results": [],
                "traversal_info": "No traversal",
                "total_results": 0,
                "vector_count": 0,
                "graph_count": 0,
                "avg_vector_similarity": 0.0,
                "avg_graph_score": 0.0,
            }
            pipeline._hybrid_retriever = mock_hr
            result = pipeline.ask_hybrid("test question")
            assert result["answer"] == "No relevant context found."

    def test_ask_with_results(self):
        """Should process results and generate answer."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()

            mock_result = MagicMock()
            mock_result.text = "FastAPI is a modern framework"
            mock_result.source = "vector"
            mock_result.similarity = 0.85
            mock_result.graph_score = 0.0
            mock_result.combined_score = 0.425

            mock_hr = MagicMock()
            mock_hr.retrieve_with_context.return_value = {
                "results": [mock_result],
                "traversal_info": "1 hop, 1 entity",
                "total_results": 1,
                "vector_count": 1,
                "graph_count": 0,
                "avg_vector_similarity": 0.85,
                "avg_graph_score": 0.0,
            }
            pipeline._hybrid_retriever = mock_hr

            mock_ranked = MagicMock()
            mock_ranked.text = "FastAPI is a modern framework"
            mock_ranked.source = "vector"
            mock_ranked.final_score = 0.85
            pipeline._reranker = MagicMock()
            pipeline._reranker.rerank.return_value = [mock_ranked]

            mock_gen = MagicMock()
            mock_gen.answer = "FastAPI is a web framework"
            mock_gen.model = "llama-3.3-70b"
            mock_gen.total_tokens = 100
            mock_gen.estimated_cost = 0.0
            pipeline._generator = MagicMock()
            pipeline._generator.generate.return_value = mock_gen

            with patch.object(HybridRetriever, "format_context", return_value="context"):
                result = pipeline.ask_hybrid("What is FastAPI?")
                assert result["answer"] == "FastAPI is a web framework"
                assert result["model"] == "llama-3.3-70b"

    def test_ask_without_sources(self):
        """Should work with show_sources=False."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()

            mock_result = MagicMock()
            mock_result.text = "test"
            mock_result.source = "vector"

            mock_hr = MagicMock()
            mock_hr.retrieve_with_context.return_value = {
                "results": [mock_result],
                "traversal_info": "",
                "total_results": 1,
                "vector_count": 1,
                "graph_count": 0,
                "avg_vector_similarity": 0.8,
                "avg_graph_score": 0.0,
            }
            pipeline._hybrid_retriever = mock_hr

            mock_ranked = MagicMock()
            mock_ranked.text = "test"
            mock_ranked.source = "vector"
            mock_ranked.final_score = 0.8
            pipeline._reranker = MagicMock()
            pipeline._reranker.rerank.return_value = [mock_ranked]

            mock_gen = MagicMock()
            mock_gen.answer = "answer"
            mock_gen.model = "test"
            mock_gen.total_tokens = 10
            mock_gen.estimated_cost = 0.0
            pipeline._generator = MagicMock()
            pipeline._generator.generate.return_value = mock_gen

            with patch.object(HybridRetriever, "format_context", return_value="ctx"):
                result = pipeline.ask_hybrid("q", show_sources=False)
                assert result["answer"] == "answer"


class TestGraphStats:
    """Tests for graph_stats method."""

    def test_graph_stats(self):
        """Should return stats dict."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            mock_gs = MagicMock()
            mock_gs.stats.return_value = {
                "total_nodes": 10,
                "total_edges": 8,
                "labels": {"Concept": 5, "Module": 5},
                "relationship_types": {"USES": 8},
                "dead_ends": 0,
            }
            pipeline._graph_store = mock_gs
            result = pipeline.graph_stats()
            assert result["total_nodes"] == 10
            assert result["total_edges"] == 8


class TestVisualize:
    """Tests for visualize method."""

    def test_visualize_full_graph(self):
        """Should export full graph."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            mock_vis = MagicMock()
            mock_vis.export_html.return_value = "./data/graph.html"
            pipeline._visualizer = mock_vis
            result = pipeline.visualize()
            assert result == "./data/graph.html"

    def test_visualize_subgraph(self):
        """Should export subgraph for entity."""
        with patch("spiral_02_graph.pipeline.VectorStore"):
            pipeline = GraphRAGPipeline()
            mock_vis = MagicMock()
            mock_vis.export_subgraph.return_value = "./data/graph_fastapi.html"
            pipeline._visualizer = mock_vis
            result = pipeline.visualize("FastAPI")
            assert result == "./data/graph_fastapi.html"


class TestMain:
    """Tests for CLI main function."""

    def test_main_no_args(self):
        """Should exit with code 1 when no args."""
        with patch("sys.argv", ["pipeline"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_unknown_command(self):
        """Should exit with code 1 for unknown command."""
        with patch("sys.argv", ["pipeline", "unknown"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_ask_no_question(self):
        """Should exit with code 1 when ask has no question."""
        with patch("sys.argv", ["pipeline", "ask"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_ingest_graph(self):
        """Should call ingest_graph."""
        with patch("sys.argv", ["pipeline", "ingest-graph"]):
            with patch.object(GraphRAGPipeline, "ingest_graph") as mock:
                mock.return_value = {}
                main()
                mock.assert_called_once_with("./data/fastapi_docs/")

    def test_main_ingest_graph_custom_path(self):
        """Should call ingest_graph with custom path."""
        with patch("sys.argv", ["pipeline", "ingest-graph", "./custom/"]):
            with patch.object(GraphRAGPipeline, "ingest_graph") as mock:
                mock.return_value = {}
                main()
                mock.assert_called_once_with("./custom/")

    def test_main_ask(self):
        """Should call ask_hybrid."""
        with patch("sys.argv", ["pipeline", "ask", "What", "is", "FastAPI?"]):
            with patch.object(GraphRAGPipeline, "ask_hybrid") as mock:
                mock.return_value = {}
                main()
                mock.assert_called_once_with("What is FastAPI?")

    def test_main_stats(self):
        """Should call graph_stats."""
        with patch("sys.argv", ["pipeline", "stats"]):
            with patch.object(GraphRAGPipeline, "graph_stats") as mock:
                mock.return_value = {}
                main()
                mock.assert_called_once()

    def test_main_visualize(self):
        """Should call visualize without entity."""
        with patch("sys.argv", ["pipeline", "visualize"]):
            with patch.object(GraphRAGPipeline, "visualize") as mock:
                mock.return_value = ""
                main()
                mock.assert_called_once_with(None)

    def test_main_visualize_entity(self):
        """Should call visualize with entity name."""
        with patch("sys.argv", ["pipeline", "visualize", "FastAPI"]):
            with patch.object(GraphRAGPipeline, "visualize") as mock:
                mock.return_value = ""
                main()
                mock.assert_called_once_with("FastAPI")
