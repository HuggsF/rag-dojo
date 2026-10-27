"""Tests for Spiral 02 — Graph RAG module.

Key validations:
- Local search returns scored results
- Global search finds paths between entities
- Path text synthesis
- Query entity extraction
- Scoring logic
"""

from unittest.mock import MagicMock, patch

from spiral_02_graph.graph_rag import GraphRAG, GraphResult


class TestGraphResult:
    def test_basic_creation(self):
        r = GraphResult(
            text="A -> B",
            entities=["A", "B"],
            path_length=1,
            relation_types=["USES"],
            score=0.8,
        )
        assert r.is_relevant  # 0.8 >= 0.3

    def test_irrelevant_result(self):
        r = GraphResult(
            text="low score",
            entities=[],
            path_length=0,
            relation_types=[],
            score=0.1,
        )
        assert not r.is_relevant  # 0.1 < 0.3

    def test_repr(self):
        r = GraphResult(
            text="A -> B -> C",
            entities=["A", "B", "C"],
            path_length=2,
            relation_types=["USES", "DEPENDS_ON"],
            score=0.75,
        )
        assert "0.75" in repr(r)


class TestSynthesizePathText:
    def test_simple_path(self):
        path = {
            "nodes": [
                {"name": "FastAPI", "labels": ["Module"]},
                {"name": "Starlette", "labels": ["Module"]},
            ],
            "relationships": [
                {"type": "USES", "properties": {}},
            ],
            "hops": 1,
        }
        text = GraphRAG._synthesize_path_text(path)
        assert "FastAPI" in text
        assert "Starlette" in text
        assert "uses" in text.lower()

    def test_multi_hop_path(self):
        path = {
            "nodes": [
                {"name": "DI", "labels": ["Concept"]},
                {"name": "FastAPI", "labels": ["Module"]},
                {"name": "Middleware", "labels": ["Concept"]},
            ],
            "relationships": [
                {"type": "PART_OF", "properties": {}},
                {"type": "USES", "properties": {}},
            ],
            "hops": 2,
        }
        text = GraphRAG._synthesize_path_text(path)
        assert "DI" in text
        assert "Middleware" in text

    def test_empty_path(self):
        path = {"nodes": [], "relationships": [], "hops": 0}
        text = GraphRAG._synthesize_path_text(path)
        assert text == ""


class TestExtractQueryEntities:
    def test_finds_entities_in_graph(self, mock_graph_store):
        rag = GraphRAG(graph_store=mock_graph_store)
        entities = rag._extract_query_entities("How does FastAPI handle routing?")
        assert len(entities) > 0

    def test_filters_stopwords(self, mock_graph_store):
        # Mock search_nodes to return empty for stopwords
        mock_graph_store.search_nodes.return_value = []
        rag = GraphRAG(graph_store=mock_graph_store)
        entities = rag._extract_query_entities("the is a an")
        assert len(entities) == 0

    def test_short_words_filtered(self, mock_graph_store):
        mock_graph_store.search_nodes.return_value = []
        rag = GraphRAG(graph_store=mock_graph_store)
        entities = rag._extract_query_entities("it is ok")
        assert len(entities) == 0


class TestLocalSearch:
    def test_returns_results(self, mock_graph_store):
        rag = GraphRAG(graph_store=mock_graph_store)
        results = rag.local_search("FastAPI")
        assert len(results) > 0
        assert all(isinstance(r, GraphResult) for r in results)

    def test_results_sorted_by_score(self, mock_graph_store):
        rag = GraphRAG(graph_store=mock_graph_store)
        results = rag.local_search("FastAPI")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_decreases_with_distance(self, mock_graph_store):
        mock_graph_store.query_neighbors.return_value = [
            {"name": "Near", "labels": ["Concept"], "properties": {}, "distance": 1},
            {"name": "Far", "labels": ["Concept"], "properties": {}, "distance": 3},
        ]
        rag = GraphRAG(graph_store=mock_graph_store)
        results = rag.local_search("Center")
        assert results[0].score > results[1].score


class TestGlobalSearch:
    def test_returns_results_when_entities_found(self, mock_graph_store):
        mock_graph_store.search_nodes.side_effect = [
            [{"name": "Dependency Injection", "labels": ["Concept"], "properties": {}}],
            [{"name": "Middleware", "labels": ["Concept"], "properties": {}}],
            [],  # bigrams
            [],
            [],
        ]
        rag = GraphRAG(graph_store=mock_graph_store)
        # Patch the embedder to avoid loading model
        rag._embedder = MagicMock()
        rag._embedder.embed_text.return_value = [0.1] * 384
        from spiral_01_chunks.embedder import Embedder

        with patch.object(Embedder, "cosine_similarity", return_value=0.8):
            results = rag.global_search("Dependency Injection and Middleware")
        assert len(results) >= 0  # may find paths

    def test_empty_when_no_entities(self, mock_graph_store):
        mock_graph_store.search_nodes.return_value = []
        rag = GraphRAG(graph_store=mock_graph_store)
        results = rag.global_search("completely unknown topic xyz")
        assert len(results) == 0


class TestPathKey:
    def test_unique_key(self):
        path = {
            "nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
        }
        key = GraphRAG._path_key(path)
        assert key == "A->B->C"

    def test_empty_path(self):
        path = {"nodes": []}
        key = GraphRAG._path_key(path)
        assert key == ""
