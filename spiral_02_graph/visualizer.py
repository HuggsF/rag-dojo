"""Visualizer — exports Neo4j graph to interactive HTML via pyvis.

Analogy for a C dev:
- This is like a memory map visualizer
- Nodes = allocated structs, colored by type
- Edges = pointers between structs
- Size = number of connections (degree centrality)
- Output = interactive HTML you can zoom, pan, and click

Optional module — nice for understanding the graph structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyvis.network import Network

from spiral_02_graph.config import GRAPH_VIS_OUTPUT
from spiral_02_graph.graph_store import GraphStore

# Color map for node labels (like memory region colors in a debugger)
LABEL_COLORS: dict[str, str] = {
    "Concept": "#4CAF50",  # green
    "Module": "#2196F3",  # blue
    "Function": "#FF9800",  # orange
    "Class": "#9C27B0",  # purple
    "Pattern": "#F44336",  # red
}
DEFAULT_COLOR = "#607D8B"  # grey for unknown labels


class GraphVisualizer:
    """Exports Neo4j graph to interactive HTML."""

    def __init__(self, graph_store: GraphStore | None = None):
        self._store = graph_store or GraphStore()

    def export_html(
        self,
        output_path: str = GRAPH_VIS_OUTPUT,
        max_nodes: int = 100,
    ) -> str:
        """Export the full graph to interactive HTML.

        Args:
            output_path: Where to save the HTML file.
            max_nodes: Maximum nodes to include (prevents huge graphs).

        Returns:
            The output file path.
        """
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
        )
        net.barnes_hut(gravity=-3000, spring_length=200)

        # Query all nodes and edges
        nodes, edges = self._fetch_graph_data(max_nodes)

        # Add nodes
        for node in nodes:
            label = node["labels"][0] if node["labels"] else "Unknown"
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            title = f"{label}: {node['name']}"
            if node.get("description"):
                title += f"\n{node['description']}"

            net.add_node(
                node["name"],
                label=node["name"],
                color=color,
                size=15 + node.get("degree", 0) * 3,
                title=title,
            )

        # Add edges
        for edge in edges:
            net.add_edge(
                edge["source"],
                edge["target"],
                label=edge["type"],
                color="#555555",
                arrows="to",
            )

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)
        return output_path

    def export_subgraph(
        self,
        entity_name: str,
        depth: int = 2,
        output_path: str | None = None,
    ) -> str:
        """Export a subgraph around a specific entity.

        Like zooming into a specific memory region and its connections.
        """
        if output_path is None:
            safe_name = entity_name.replace(" ", "_").lower()
            output_path = f"./data/graph_{safe_name}.html"

        net = Network(
            height="750px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
        )
        net.barnes_hut(gravity=-3000, spring_length=200)

        # Get neighbors
        neighbors = self._store.query_neighbors(entity_name, depth=depth, limit=50)

        # Add center node
        net.add_node(
            entity_name,
            label=entity_name,
            color="#FFD700",  # gold for center
            size=30,
            title=f"Center: {entity_name}",
        )

        # Add neighbor nodes
        for n in neighbors:
            label = n["labels"][0] if n["labels"] else "Unknown"
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            net.add_node(
                n["name"],
                label=n["name"],
                color=color,
                size=20,
                title=f"{label}: {n['name']} ({n['distance']} hops)",
            )
            # Add edge (simplified — just connected)
            net.add_edge(entity_name, n["name"], color="#555555", arrows="to")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)
        return output_path

    def _fetch_graph_data(
        self,
        max_nodes: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch nodes and edges from Neo4j."""
        with self._store._driver.session() as session:
            # Nodes with degree
            node_result = session.run(
                "MATCH (n) "
                "OPTIONAL MATCH (n)-[r]-() "
                "WITH n, count(r) as degree "
                "RETURN n.name as name, labels(n) as labels, "
                "n.description as description, degree "
                "ORDER BY degree DESC "
                "LIMIT $limit",
                limit=max_nodes,
            )
            nodes = [
                {
                    "name": r["name"],
                    "labels": r["labels"],
                    "description": r["description"] or "",
                    "degree": r["degree"],
                }
                for r in node_result
            ]

            # Get node names for edge filtering
            node_names = {n["name"] for n in nodes}

            # Edges between included nodes
            edge_result = session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE a.name IN $names AND b.name IN $names "
                "RETURN a.name as source, b.name as target, type(r) as type",
                names=list(node_names),
            )
            edges = [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                }
                for r in edge_result
            ]

        return nodes, edges

    @staticmethod
    def _label_to_color(label: str) -> str:
        """Map node label to color."""
        return LABEL_COLORS.get(label, DEFAULT_COLOR)
