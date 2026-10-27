"""Tests for Spiral 02 — Graph Store module.

Unit tests mock the Neo4j driver.
Integration tests (marked @pytest.mark.integration) need a running Neo4j instance.

Key validations:
- Node and edge persistence (MERGE semantics)
- Path queries
- Neighbor queries
- Dead-end marking
- Stats and clear
- Label/relation type validation
"""

from unittest.mock import MagicMock, patch

import pytest

from spiral_02_graph.entity_extractor import Entity, Relation
from spiral_02_graph.graph_store import GraphStore, _validate_label, _validate_relation_type


class TestValidation:
    def test_valid_label(self):
        assert _validate_label("Concept") == "Concept"
        assert _validate_label("Module") == "Module"
        assert _validate_label("Function") == "Function"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Invalid label"):
            _validate_label("BadLabel")

    def test_valid_relation_type(self):
        assert _validate_relation_type("USES") == "USES"
        assert _validate_relation_type("DEPENDS_ON") == "DEPENDS_ON"

    def test_invalid_relation_type_raises(self):
        with pytest.raises(ValueError, match="Invalid relation type"):
            _validate_relation_type("INVALID_TYPE")


class TestGraphStoreMocked:
    """Unit tests with mocked Neo4j driver."""

    @pytest.fixture
    def store(self):
        """Create GraphStore with mocked driver."""
        with patch("spiral_02_graph.graph_store.GraphDatabase") as mock_gdb:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity.return_value = None
            mock_gdb.driver.return_value = mock_driver

            store = GraphStore(uri="bolt://fake:7687", user="neo4j", password="test")
            yield store, mock_driver

    def test_construction(self, store):
        graph_store, mock_driver = store
        mock_driver.verify_connectivity.assert_called_once()

    def test_persist_node(self, store):
        graph_store, mock_driver = store
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        entity = Entity(name="FastAPI", label="Module")
        graph_store.persist_node(entity)

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MERGE" in call_args[0][0]
        assert "Module" in call_args[0][0]
        assert call_args[1]["name"] == "FastAPI"

    def test_persist_edge(self, store):
        graph_store, mock_driver = store
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        relation = Relation(source="FastAPI", target="Starlette", relation_type="USES")
        graph_store.persist_edge(relation)

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH" in call_args[0][0]
        assert "MERGE" in call_args[0][0]
        assert "USES" in call_args[0][0]

    def test_persist_extraction(self, store, sample_extraction_result):
        graph_store, mock_driver = store
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        graph_store.persist_extraction(sample_extraction_result)

        # Should have called run() for each entity + each relation
        expected_calls = len(sample_extraction_result.entities) + len(
            sample_extraction_result.relations
        )
        assert mock_session.run.call_count == expected_calls

    def test_mark_dead_end(self, store):
        graph_store, mock_driver = store
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        graph_store.mark_dead_end("A", "B", "USES")

        call_args = mock_session.run.call_args
        assert "dead_end" in call_args[0][0]

    def test_clear(self, store):
        graph_store, mock_driver = store
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        graph_store.clear()

        call_args = mock_session.run.call_args
        assert "DETACH DELETE" in call_args[0][0]

    def test_context_manager(self, store):
        graph_store, mock_driver = store
        with graph_store:
            pass
        mock_driver.close.assert_called_once()

    def test_close(self, store):
        graph_store, mock_driver = store
        graph_store.close()
        mock_driver.close.assert_called_once()

    def test_connectivity_error(self):
        """Test that ServiceUnavailable is converted to ConnectionError."""
        from neo4j.exceptions import ServiceUnavailable

        with patch("spiral_02_graph.graph_store.GraphDatabase") as mock_gdb:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity.side_effect = ServiceUnavailable("down")
            mock_gdb.driver.return_value = mock_driver

            with pytest.raises(ConnectionError, match="Cannot connect to Neo4j"):
                GraphStore(uri="bolt://fake:7687", user="neo4j", password="test")

    def _make_session(self, mock_driver):
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return mock_session

    def test_query_path(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_node_a = MagicMock()
        mock_node_a.get.return_value = "FastAPI"
        mock_node_a.labels = frozenset(["Module"])
        mock_node_a.__iter__ = lambda s: iter([("name", "FastAPI")])
        mock_node_a.items.return_value = [("name", "FastAPI")]
        mock_node_a.keys.return_value = ["name"]
        mock_node_a.__getitem__ = lambda s, k: "FastAPI"

        mock_node_b = MagicMock()
        mock_node_b.get.return_value = "Middleware"
        mock_node_b.labels = frozenset(["Concept"])
        mock_node_b.__iter__ = lambda s: iter([("name", "Middleware")])
        mock_node_b.items.return_value = [("name", "Middleware")]
        mock_node_b.keys.return_value = ["name"]
        mock_node_b.__getitem__ = lambda s, k: "Middleware"

        mock_rel = MagicMock()
        mock_rel.type = "USES"
        mock_rel.__iter__ = lambda s: iter([])
        mock_rel.items.return_value = []
        mock_rel.keys.return_value = []

        mock_path = MagicMock()
        mock_path.nodes = [mock_node_a, mock_node_b]
        mock_path.relationships = [mock_rel]

        mock_record = {"path": mock_path, "hops": 1}
        mock_session.run.return_value = [mock_record]

        paths = graph_store.query_path("FastAPI", "Middleware")
        assert len(paths) == 1
        assert paths[0]["hops"] == 1
        assert len(paths[0]["nodes"]) == 2

    def test_query_neighbors(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_session.run.return_value = [
            {"name": "Middleware", "labels": ["Concept"], "properties": {}, "distance": 1},
            {"name": "Starlette", "labels": ["Module"], "properties": {}, "distance": 2},
        ]

        neighbors = graph_store.query_neighbors("FastAPI", depth=2, limit=10)
        assert len(neighbors) == 2
        assert neighbors[0]["name"] == "Middleware"
        assert neighbors[1]["distance"] == 2

    def test_query_by_label(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_session.run.return_value = [
            {"name": "FastAPI", "properties": {"desc": "framework"}},
            {"name": "Starlette", "properties": {}},
        ]

        results = graph_store.query_by_label("Module", limit=5)
        assert len(results) == 2
        assert results[0]["name"] == "FastAPI"

    def test_search_nodes(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_session.run.return_value = [
            {"name": "FastAPI", "labels": ["Module"], "properties": {}},
        ]

        results = graph_store.search_nodes("fast", limit=5)
        assert len(results) == 1
        assert results[0]["name"] == "FastAPI"
        call_args = mock_session.run.call_args
        assert call_args[1]["search_term"] == "fast"

    def test_is_dead_end_true(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_result = MagicMock()
        mock_result.single.return_value = {"dead_end": True}
        mock_session.run.return_value = mock_result

        assert graph_store.is_dead_end("A", "B", "USES") is True

    def test_is_dead_end_false_no_record(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        assert graph_store.is_dead_end("A", "B", "USES") is False

    def test_node_count(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_result = MagicMock()
        mock_result.single.return_value = {"cnt": 42}
        mock_session.run.return_value = mock_result

        assert graph_store.node_count == 42

    def test_edge_count(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        mock_result = MagicMock()
        mock_result.single.return_value = {"cnt": 100}
        mock_session.run.return_value = mock_result

        assert graph_store.edge_count == 100

    def test_stats(self, store):
        graph_store, mock_driver = store
        mock_session = self._make_session(mock_driver)

        # stats() makes multiple session.run calls
        label_result = [{"label": "Concept", "cnt": 10}, {"label": "Module", "cnt": 5}]
        rel_result = [{"type": "USES", "cnt": 8}]
        dead_end_result = MagicMock()
        dead_end_result.single.return_value = {"cnt": 2}
        node_count_result = MagicMock()
        node_count_result.single.return_value = {"cnt": 15}
        edge_count_result = MagicMock()
        edge_count_result.single.return_value = {"cnt": 8}

        mock_session.run.side_effect = [
            label_result,
            rel_result,
            dead_end_result,
            node_count_result,
            edge_count_result,
        ]

        stats = graph_store.stats()
        assert stats["labels"] == {"Concept": 10, "Module": 5}
        assert stats["relationship_types"] == {"USES": 8}
        assert stats["dead_ends"] == 2
        assert stats["total_nodes"] == 15
        assert stats["total_edges"] == 8
