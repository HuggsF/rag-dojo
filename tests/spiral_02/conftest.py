"""Shared fixtures for Spiral 02 tests."""

from unittest.mock import MagicMock

import pytest

from spiral_02_graph.entity_extractor import Entity, ExtractionResult, Relation
from spiral_02_graph.graph_rag import GraphResult
from spiral_02_graph.hybrid_retriever import HybridResult


@pytest.fixture
def sample_entities() -> list[Entity]:
    """Sample entities from FastAPI domain."""
    return [
        Entity(name="FastAPI", label="Module", properties={"description": "Modern web framework"}),
        Entity(
            name="Dependency Injection", label="Concept", properties={"description": "DI pattern"}
        ),
        Entity(
            name="Middleware",
            label="Concept",
            properties={"description": "Request processing layer"},
        ),
        Entity(name="Starlette", label="Module", properties={"description": "ASGI framework"}),
        Entity(name="APIRouter", label="Class", properties={"description": "Route grouping"}),
        Entity(name="Depends", label="Function", properties={"description": "DI function"}),
    ]


@pytest.fixture
def sample_relations() -> list[Relation]:
    """Sample relations between FastAPI entities."""
    return [
        Relation(source="FastAPI", target="Starlette", relation_type="USES"),
        Relation(source="FastAPI", target="Middleware", relation_type="USES"),
        Relation(source="Depends", target="Dependency Injection", relation_type="PART_OF"),
        Relation(source="Middleware", target="FastAPI", relation_type="DEPENDS_ON"),
        Relation(source="APIRouter", target="FastAPI", relation_type="EXTENDS"),
    ]


@pytest.fixture
def sample_extraction_result(
    sample_entities: list[Entity],
    sample_relations: list[Relation],
) -> ExtractionResult:
    """Sample extraction result."""
    return ExtractionResult(
        entities=sample_entities,
        relations=sample_relations,
        chunk_id="fastapi.md::chunk_0",
        model="llama-3.3-70b-versatile",
        input_tokens=500,
        output_tokens=200,
    )


@pytest.fixture
def sample_graph_results() -> list[GraphResult]:
    """Sample graph retrieval results."""
    return [
        GraphResult(
            text="Dependency Injection -> part of -> FastAPI -> uses -> Middleware",
            entities=["Dependency Injection", "FastAPI", "Middleware"],
            path_length=2,
            relation_types=["PART_OF", "USES"],
            score=0.85,
        ),
        GraphResult(
            text="FastAPI -> uses -> Starlette",
            entities=["FastAPI", "Starlette"],
            path_length=1,
            relation_types=["USES"],
            score=0.72,
        ),
    ]


@pytest.fixture
def sample_hybrid_results() -> list[HybridResult]:
    """Sample hybrid retrieval results."""
    return [
        HybridResult(
            text="FastAPI is a modern web framework for building APIs.",
            source="hybrid",
            similarity=0.85,
            graph_score=0.78,
            combined_score=0.815,
            path_info="FastAPI -> uses -> Middleware",
        ),
        HybridResult(
            text="Dependency Injection allows you to declare dependencies.",
            source="vector",
            similarity=0.72,
            graph_score=0.0,
            combined_score=0.36,
        ),
        HybridResult(
            text="Middleware processes requests before they reach the endpoint.",
            source="graph",
            similarity=0.0,
            graph_score=0.65,
            combined_score=0.325,
            path_info="Middleware -> depends_on -> FastAPI",
        ),
    ]


@pytest.fixture
def mock_graph_store():
    """Mock GraphStore that doesn't require Neo4j."""
    store = MagicMock()
    store.node_count = 6
    store.edge_count = 5
    store.search_nodes.return_value = [
        {"name": "FastAPI", "labels": ["Module"], "properties": {}},
    ]
    store.query_neighbors.return_value = [
        {"name": "Middleware", "labels": ["Concept"], "properties": {}, "distance": 1},
        {"name": "Starlette", "labels": ["Module"], "properties": {}, "distance": 1},
    ]
    store.query_path.return_value = [
        {
            "nodes": [
                {"name": "Dependency Injection", "labels": ["Concept"]},
                {"name": "FastAPI", "labels": ["Module"]},
                {"name": "Middleware", "labels": ["Concept"]},
            ],
            "relationships": [
                {"type": "PART_OF", "properties": {}},
                {"type": "USES", "properties": {}},
            ],
            "hops": 2,
        }
    ]
    store.stats.return_value = {
        "total_nodes": 6,
        "total_edges": 5,
        "labels": {"Concept": 2, "Module": 2, "Function": 1, "Class": 1},
        "relationship_types": {"USES": 2, "PART_OF": 1, "DEPENDS_ON": 1, "EXTENDS": 1},
        "dead_ends": 0,
    }
    return store
