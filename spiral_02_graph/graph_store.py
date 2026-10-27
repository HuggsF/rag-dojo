"""Graph Store — Neo4j wrapper for knowledge graph persistence.

Analogy for a C dev:
- Neo4j is a graph-structured heap
- Nodes are structs with labels (like tagged unions)
- Edges are pointers with types and properties
- Cypher is the query language (like SQL but for graphs)
- MERGE is like get_or_create (idempotent allocation)
- shortestPath is BFS with a depth limit

This follows the same pattern as spiral_01's VectorStore:
- Constructor takes connection params with defaults from config
- CRUD methods for nodes and edges
- Stats and clear for maintenance
- Context manager support for clean resource handling
"""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from spiral_02_graph.config import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONS,
    GRAPH_SEARCH_DEPTH,
    GRAPH_TOP_K,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)
from spiral_02_graph.entity_extractor import Entity, ExtractionResult, Relation


def _validate_label(label: str) -> str:
    """Validate a node label against the allowlist.

    Neo4j doesn't support parameterized labels, so we must validate
    before interpolating into Cypher. Like bounds-checking before
    writing to a buffer.
    """
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Invalid label '{label}'. Allowed: {ALLOWED_LABELS}")
    return label


def _validate_relation_type(relation_type: str) -> str:
    """Validate a relationship type against the allowlist."""
    if relation_type not in ALLOWED_RELATIONS:
        raise ValueError(f"Invalid relation type '{relation_type}'. Allowed: {ALLOWED_RELATIONS}")
    return relation_type


class GraphStore:
    """Neo4j wrapper for storing and querying the knowledge graph."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity on creation
        try:
            self._driver.verify_connectivity()
        except ServiceUnavailable as e:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {uri}. "
                "The canonical Neo4j runs in K8s (pod: neo4j, svc: neo4j). "
                "From rag-test pod: bolt://neo4j:7687. "
                "From Windows local: kubectl port-forward svc/neo4j 7687:7687, "
                "then set NEO4J_URI=bolt://localhost:7687"
            ) from e

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    # --- Write operations ---

    def persist_node(self, entity: Entity) -> None:
        """Create or merge a node in the graph.

        Uses MERGE to avoid duplicates (idempotent like realloc on same pointer).
        """
        label = _validate_label(entity.label)
        with self._driver.session() as session:
            session.run(
                f"MERGE (n:{label} {{name: $name}}) "
                "SET n += $properties, n.source_chunk_id = $source_chunk_id",
                name=entity.name,
                properties=entity.properties,
                source_chunk_id=entity.source_chunk_id,
            )

    def persist_edge(self, relation: Relation) -> None:
        """Create or merge an edge between two nodes.

        Matches nodes by name (any label) and creates a typed relationship.
        """
        rel_type = _validate_relation_type(relation.relation_type)
        with self._driver.session() as session:
            session.run(
                f"MATCH (a {{name: $source}}), (b {{name: $target}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += $properties, r.source_chunk_id = $source_chunk_id",
                source=relation.source,
                target=relation.target,
                properties=relation.properties,
                source_chunk_id=relation.source_chunk_id,
            )

    def persist_extraction(self, result: ExtractionResult) -> None:
        """Persist all entities and relations from an extraction result.

        Nodes first, then edges (edges reference existing nodes).
        """
        for entity in result.entities:
            self.persist_node(entity)
        for relation in result.relations:
            self.persist_edge(relation)

    # --- Read operations ---

    def query_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = GRAPH_SEARCH_DEPTH,
    ) -> list[dict[str, Any]]:
        """Find shortest path(s) between two entities.

        Like BFS with a depth limit — finds the optimal route through the graph.
        Returns list of path dicts with nodes and relationships.
        """
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a {name: $start}), (b {name: $end}), "
                f"path = shortestPath((a)-[*..{max_depth}]-(b)) "
                "RETURN path, length(path) as hops",
                start=start_entity,
                end=end_entity,
            )
            paths = []
            for record in result:
                path = record["path"]
                nodes = [
                    {
                        "name": node.get("name", ""),
                        "labels": list(node.labels),
                        "properties": dict(node),
                    }
                    for node in path.nodes
                ]
                relationships = [
                    {
                        "type": rel.type,
                        "properties": dict(rel),
                    }
                    for rel in path.relationships
                ]
                paths.append(
                    {
                        "nodes": nodes,
                        "relationships": relationships,
                        "hops": record["hops"],
                    }
                )
            return paths

    def query_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        limit: int = GRAPH_TOP_K,
    ) -> list[dict[str, Any]]:
        """Get neighbors within N hops of an entity.

        Like exploring adjacent memory blocks — find what's connected.
        """
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (n {{name: $name}})-[r*1..{depth}]-(m) "
                "WHERE NOT coalesce(r[-1].dead_end, false) "
                "RETURN DISTINCT m.name as name, labels(m) as labels, "
                "properties(m) as properties, "
                "size(r) as distance "
                "ORDER BY distance ASC "
                "LIMIT $limit",
                name=entity_name,
                limit=limit,
            )
            return [
                {
                    "name": record["name"],
                    "labels": record["labels"],
                    "properties": record["properties"],
                    "distance": record["distance"],
                }
                for record in result
            ]

    def query_by_label(self, label: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get all nodes of a specific label."""
        label = _validate_label(label)
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (n:{label}) RETURN n.name as name, properties(n) as properties "
                "LIMIT $limit",
                limit=limit,
            )
            return [
                {"name": record["name"], "properties": record["properties"]} for record in result
            ]

    def search_nodes(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Text search on node names (case-insensitive CONTAINS).

        Like grep on struct names — fuzzy match for entity lookup.
        """
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($search_term) "
                "RETURN n.name as name, labels(n) as labels, "
                "properties(n) as properties "
                "LIMIT $limit",
                search_term=query,
                limit=limit,
            )
            return [
                {
                    "name": record["name"],
                    "labels": record["labels"],
                    "properties": record["properties"],
                }
                for record in result
            ]

    # --- Dead-end marking (bonus challenge) ---

    def mark_dead_end(self, source: str, target: str, relation_type: str) -> None:
        """Mark an edge as a dead end (similarity < threshold).

        Like marking a memory region as unmapped — future traversals skip it.
        """
        rel_type = _validate_relation_type(relation_type)
        with self._driver.session() as session:
            session.run(
                f"MATCH (a {{name: $source}})-[r:{rel_type}]->(b {{name: $target}}) "
                "SET r.dead_end = true",
                source=source,
                target=target,
            )

    def is_dead_end(self, source: str, target: str, relation_type: str) -> bool:
        """Check if an edge is marked as dead end."""
        rel_type = _validate_relation_type(relation_type)
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (a {{name: $source}})-[r:{rel_type}]->(b {{name: $target}}) "
                "RETURN coalesce(r.dead_end, false) as dead_end",
                source=source,
                target=target,
            )
            record = result.single()
            return bool(record["dead_end"]) if record else False

    # --- Stats & maintenance ---

    @property
    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        with self._driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as cnt")
            record = result.single()
            return record["cnt"] if record else 0  # type: ignore[no-any-return]

    @property
    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        with self._driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as cnt")
            record = result.single()
            return record["cnt"] if record else 0  # type: ignore[no-any-return]

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        with self._driver.session() as session:
            # Label distribution
            label_result = session.run(
                "MATCH (n) UNWIND labels(n) as label "
                "RETURN label, count(*) as cnt ORDER BY cnt DESC"
            )
            labels = {r["label"]: r["cnt"] for r in label_result}

            # Relationship type distribution
            rel_result = session.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(*) as cnt ORDER BY cnt DESC"
            )
            rel_types = {r["type"]: r["cnt"] for r in rel_result}

            # Dead-end count
            dead_end_result = session.run(
                "MATCH ()-[r]->() WHERE r.dead_end = true RETURN count(r) as cnt"
            )
            dead_end_record = dead_end_result.single()
            dead_ends = dead_end_record["cnt"] if dead_end_record else 0  # type: ignore[index]

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "labels": labels,
            "relationship_types": rel_types,
            "dead_ends": dead_ends,
        }

    def clear(self) -> None:
        """Delete all nodes and relationships. Like free() on the entire heap."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
