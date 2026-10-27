"""Tests for spiral_02_graph.visualizer — graph visualization."""

from unittest.mock import MagicMock, patch

from spiral_02_graph.visualizer import (
    DEFAULT_COLOR,
    LABEL_COLORS,
    GraphVisualizer,
)


class TestLabelColors:
    """Tests for color constants."""

    def test_known_labels_have_colors(self):
        """All expected labels should have colors."""
        assert "Concept" in LABEL_COLORS
        assert "Module" in LABEL_COLORS
        assert "Function" in LABEL_COLORS
        assert "Class" in LABEL_COLORS
        assert "Pattern" in LABEL_COLORS

    def test_default_color_exists(self):
        """Default color should be defined."""
        assert DEFAULT_COLOR == "#607D8B"

    def test_label_to_color_known(self):
        """_label_to_color should return mapped color."""
        assert GraphVisualizer._label_to_color("Concept") == "#4CAF50"

    def test_label_to_color_unknown(self):
        """_label_to_color should return default for unknown labels."""
        assert GraphVisualizer._label_to_color("UnknownLabel") == DEFAULT_COLOR


class TestGraphVisualizerInit:
    """Tests for visualizer initialization."""

    def test_init_with_store(self):
        """Should accept a graph store."""
        mock_store = MagicMock()
        viz = GraphVisualizer(mock_store)
        assert viz._store is mock_store

    def test_init_without_store(self):
        """Should create default graph store."""
        with patch("spiral_02_graph.visualizer.GraphStore") as MockGS:
            mock_gs = MagicMock()
            MockGS.return_value = mock_gs
            viz = GraphVisualizer()
            assert viz._store is mock_gs


class TestExportHtml:
    """Tests for export_html method."""

    @patch("spiral_02_graph.visualizer.Network")
    @patch("spiral_02_graph.visualizer.Path")
    def test_export_html_basic(self, mock_path, mock_network_cls):
        """Should create HTML with nodes and edges."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_store._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock node query
        mock_session.run.side_effect = [
            [
                {
                    "name": "FastAPI",
                    "labels": ["Module"],
                    "description": "Web framework",
                    "degree": 3,
                }
            ],
            [{"source": "FastAPI", "target": "Starlette", "type": "USES"}],
        ]

        viz = GraphVisualizer(mock_store)
        mock_net = MagicMock()
        mock_network_cls.return_value = mock_net

        result = viz.export_html("/tmp/test.html")
        assert result == "/tmp/test.html"
        mock_net.add_node.assert_called_once()
        mock_net.save_graph.assert_called_once()

    @patch("spiral_02_graph.visualizer.Network")
    @patch("spiral_02_graph.visualizer.Path")
    def test_export_html_no_description(self, mock_path, mock_network_cls):
        """Should handle nodes without description."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_store._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_session.run.side_effect = [
            [{"name": "Test", "labels": [], "description": None, "degree": 0}],
            [],
        ]

        viz = GraphVisualizer(mock_store)
        mock_net = MagicMock()
        mock_network_cls.return_value = mock_net

        viz.export_html("/tmp/test.html")
        mock_net.add_node.assert_called_once()


class TestExportSubgraph:
    """Tests for export_subgraph method."""

    @patch("spiral_02_graph.visualizer.Network")
    @patch("spiral_02_graph.visualizer.Path")
    def test_export_subgraph(self, mock_path, mock_network_cls):
        """Should create subgraph around entity."""
        mock_store = MagicMock()
        mock_store.query_neighbors.return_value = [
            {"name": "Middleware", "labels": ["Concept"], "distance": 1},
        ]

        viz = GraphVisualizer(mock_store)
        mock_net = MagicMock()
        mock_network_cls.return_value = mock_net

        result = viz.export_subgraph("FastAPI")
        assert "fastapi" in result
        # Center node + 1 neighbor
        assert mock_net.add_node.call_count == 2
        mock_net.add_edge.assert_called_once()

    @patch("spiral_02_graph.visualizer.Network")
    @patch("spiral_02_graph.visualizer.Path")
    def test_export_subgraph_custom_path(self, mock_path, mock_network_cls):
        """Should use custom output path."""
        mock_store = MagicMock()
        mock_store.query_neighbors.return_value = []

        viz = GraphVisualizer(mock_store)
        mock_net = MagicMock()
        mock_network_cls.return_value = mock_net

        result = viz.export_subgraph("Test", output_path="/tmp/custom.html")
        assert result == "/tmp/custom.html"

    @patch("spiral_02_graph.visualizer.Network")
    @patch("spiral_02_graph.visualizer.Path")
    def test_export_subgraph_no_neighbors(self, mock_path, mock_network_cls):
        """Should handle entity with no neighbors."""
        mock_store = MagicMock()
        mock_store.query_neighbors.return_value = []

        viz = GraphVisualizer(mock_store)
        mock_net = MagicMock()
        mock_network_cls.return_value = mock_net

        viz.export_subgraph("Isolated")
        # Only center node
        mock_net.add_node.assert_called_once()
        mock_net.add_edge.assert_not_called()
