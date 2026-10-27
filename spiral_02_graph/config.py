"""Configuration for Spiral 02 — Graph RAG settings.

Analogy for a C dev:
- NEO4J_URI = database connection string (like a socket address)
- GRAPH_SEARCH_DEPTH = maximum hops in traversal (like BFS depth limit)
- VECTOR_WEIGHT/GRAPH_WEIGHT = blending factors for hybrid search
- DEAD_END_THRESHOLD = minimum quality bar before marking a path as useless
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Import shared LLM config from spiral_01
from spiral_01_chunks.config import (  # noqa: E402, F401
    GROQ_API_KEY,
    GROQ_API_KEYS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

# Neo4j connection
# IMPORTANT: The canonical Neo4j runs inside Kubernetes (pod: neo4j, svc: neo4j).
# From inside K8s (rag-test pod): bolt://neo4j:7687
# From Windows local (dev/test only): bolt://localhost:7687 via port-forward
# Default points to K8s service name — override via NEO4J_URI env var if needed.
NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")

# Entity extraction
EXTRACTION_MODEL: str = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_ENTITIES_PER_CHUNK: int = int(os.environ.get("MAX_ENTITIES_PER_CHUNK", "10"))
MAX_RELATIONS_PER_CHUNK: int = int(os.environ.get("MAX_RELATIONS_PER_CHUNK", "15"))

# Graph RAG
GRAPH_SEARCH_DEPTH: int = int(os.environ.get("GRAPH_SEARCH_DEPTH", "3"))
GRAPH_TOP_K: int = int(os.environ.get("GRAPH_TOP_K", "5"))

# Hybrid retrieval weights
VECTOR_WEIGHT: float = float(os.environ.get("VECTOR_WEIGHT", "0.5"))
GRAPH_WEIGHT: float = float(os.environ.get("GRAPH_WEIGHT", "0.5"))

# Dead-end marking (bonus challenge)
DEAD_END_THRESHOLD: float = float(os.environ.get("DEAD_END_THRESHOLD", "0.3"))

# Reranker
RERANKER_MODEL: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_CROSS_ENCODER: bool = os.environ.get("USE_CROSS_ENCODER", "false").lower() == "true"

# Visualization
GRAPH_VIS_OUTPUT: str = os.environ.get("GRAPH_VIS_OUTPUT", "./data/graph_vis.html")

# Allowed node labels (validated before Cypher interpolation)
ALLOWED_LABELS: set[str] = {"Concept", "Module", "Function", "Class", "Pattern"}

# Allowed relationship types
ALLOWED_RELATIONS: set[str] = {
    "USES",
    "DEPENDS_ON",
    "PART_OF",
    "RELATED_TO",
    "IMPLEMENTS",
    "EXTENDS",
}
