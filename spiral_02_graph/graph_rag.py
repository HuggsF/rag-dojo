"""Graph RAG — two-blade graph search: local + global.

Analogy for a C dev:
- local_search = reading adjacent memory blocks (neighbors within N hops)
- global_search = BFS/DFS across the entire graph to find connections
- GPS (vector) finds the nearest point. Waze (graph) finds the best route.

Two blades:
1. Local search: "What's near this entity?" (neighborhood exploration)
2. Global search: "How does A connect to B?" (path finding)

Both return GraphResult objects scored by relevance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from spiral_01_chunks.embedder import Embedder, get_embedder
from spiral_02_graph.config import (
    DEAD_END_THRESHOLD,
    GRAPH_SEARCH_DEPTH,
    GRAPH_TOP_K,
)
from spiral_02_graph.graph_store import GraphStore

# Common stopwords to filter when extracting query entities
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "and",
    "but",
    "or",
    "if",
    "while",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    # Portuguese stopwords (project is bilingual)
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "sem",
    "sob",
    "sobre",
    "entre",
    "qual",
    "como",
    "que",
    "e",
}


@dataclass
class GraphResult:
    """A single graph retrieval result."""

    text: str  # synthesized text from graph path
    entities: list[str]  # entity names involved
    path_length: int  # number of hops
    relation_types: list[str]  # relationship types traversed
    score: float  # graph relevance score
    source: str = "graph"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_relevant(self) -> bool:
        return self.score >= DEAD_END_THRESHOLD

    def __repr__(self) -> str:
        ents = " -> ".join(self.entities[:3])
        return f"GraphResult(score={self.score:.3f}, hops={self.path_length}, '{ents}')"


class GraphRAG:
    """Two-blade graph search: local neighborhood + global path finding."""

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        embedder: Embedder | None = None,
    ):
        self._store = graph_store or GraphStore()
        self._embedder = embedder or get_embedder()

    def local_search(
        self,
        entity_name: str,
        depth: int = GRAPH_SEARCH_DEPTH,
        top_k: int = GRAPH_TOP_K,
    ) -> list[GraphResult]:
        """Search neighborhood of an entity.

        Like reading adjacent struct members — what's directly connected?

        Args:
            entity_name: The entity to search around.
            depth: Maximum hops from the entity.
            top_k: Maximum results to return.

        Returns:
            List of GraphResult, scored by proximity (closer = higher score).
        """
        neighbors = self._store.query_neighbors(entity_name, depth=depth, limit=top_k)

        results: list[GraphResult] = []
        for neighbor in neighbors:
            distance = neighbor.get("distance", 1)
            # Score: closer neighbors score higher (inverse distance)
            score = 1.0 / (1 + distance)
            text = self._synthesize_neighbor_text(entity_name, neighbor)
            results.append(
                GraphResult(
                    text=text,
                    entities=[entity_name, neighbor["name"]],
                    path_length=distance,
                    relation_types=[],  # simplified for neighbor search
                    score=round(score, 4),
                    metadata=neighbor.get("properties", {}),
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def global_search(
        self,
        query: str,
        top_k: int = GRAPH_TOP_K,
    ) -> list[GraphResult]:
        """Search across the graph for paths relevant to the query.

        Like running BFS between two endpoints — find how concepts connect.

        Strategy:
        1. Extract key entity names from the query
        2. Find matching entities in the graph
        3. Find paths between all pairs
        4. Score paths by embedding similarity to the query
        """
        # Extract entities from query
        query_entities = self._extract_query_entities(query)
        if len(query_entities) < 2:
            # Not enough entities for path finding, fall back to local search
            if query_entities:
                return self.local_search(query_entities[0], top_k=top_k)
            return []

        # Find paths between all pairs of extracted entities
        results: list[GraphResult] = []
        seen_paths: set[str] = set()

        for i, start in enumerate(query_entities):
            for end in query_entities[i + 1 :]:
                paths = self._store.query_path(start, end)
                for path in paths:
                    path_key = self._path_key(path)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    path_text = self._synthesize_path_text(path)
                    score = self._score_path(path_text, query)

                    entity_names = [n["name"] for n in path["nodes"]]
                    rel_types = [r["type"] for r in path["relationships"]]

                    results.append(
                        GraphResult(
                            text=path_text,
                            entities=entity_names,
                            path_length=path["hops"],
                            relation_types=rel_types,
                            score=round(score, 4),
                        )
                    )

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity names from a query by matching against the graph.

        Strategy: tokenize query, remove stopwords, search graph for matches.
        Like parsing tokens from a command line and resolving them against a symbol table.
        """
        # Tokenize and clean
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)
        significant_words = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]

        # Also try multi-word phrases (bigrams)
        phrases: list[str] = list(significant_words)
        for i in range(len(significant_words) - 1):
            phrases.append(f"{significant_words[i]} {significant_words[i + 1]}")

        # Search graph for each term
        found_entities: list[str] = []
        seen: set[str] = set()

        for term in phrases:
            matches = self._store.search_nodes(term, limit=3)
            for match in matches:
                name = match["name"]
                if name not in seen:
                    seen.add(name)
                    found_entities.append(name)

        return found_entities

    @staticmethod
    def _synthesize_path_text(path: dict) -> str:
        """Convert a graph path into readable text.

        Input: {"nodes": [...], "relationships": [...], "hops": N}
        Output: "Entity A depends on Entity B, which uses Entity C."
        """
        nodes = path["nodes"]
        rels = path["relationships"]

        if not nodes:
            return ""

        parts: list[str] = [nodes[0]["name"]]
        for i, rel in enumerate(rels):
            rel_type = rel["type"].lower().replace("_", " ")
            target = nodes[i + 1]["name"] if i + 1 < len(nodes) else "?"
            parts.append(f"{rel_type} {target}")

        return " -> ".join(parts)

    @staticmethod
    def _synthesize_neighbor_text(center: str, neighbor: dict) -> str:
        """Synthesize text for a neighbor relationship."""
        name = neighbor.get("name", "?")
        labels = neighbor.get("labels", [])
        label = labels[0] if labels else "Entity"
        distance = neighbor.get("distance", 1)
        return f"{center} is connected to {name} ({label}, {distance} hop(s) away)"

    def _score_path(self, path_text: str, query: str) -> float:
        """Score a path's relevance to the query using embedding similarity."""
        path_vec = self._embedder.embed_text(path_text)
        query_vec = self._embedder.embed_text(query)
        return max(0.0, Embedder.cosine_similarity(path_vec, query_vec))

    @staticmethod
    def _path_key(path: dict) -> str:
        """Create a unique key for a path to avoid duplicates."""
        names = [n["name"] for n in path.get("nodes", [])]
        return "->".join(names)
