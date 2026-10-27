"""Configuration for Spiral 01 — all knobs in one place.

Think of this as your compiler flags:
- CHUNK_SIZE = block size (malloc granularity)
- CHUNK_OVERLAP = overlap between blocks (context preservation)
- TOP_K = how many results to return (like LIMIT in SQL)
- EMBEDDING_MODEL = which hash function for semantic space
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Chunking
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "128"))

# Embedding
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS: int = 384  # MiniLM-L6-v2 output size

# Retrieval
TOP_K: int = int(os.environ.get("TOP_K", "7"))
SIMILARITY_THRESHOLD: float = 0.3  # minimum cosine similarity to consider relevant

# ChromaDB
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME: str = os.environ.get("CHROMA_COLLECTION_NAME", "rag_dojo_spiral_01")

# LLM (Groq — free tier)
# GROQ_API_KEYS: comma-separated list for round-robin rotation (3x rate limit)
# Falls back to GROQ_API_KEY (single key) if GROQ_API_KEYS is not set.
_raw_keys = os.environ.get("GROQ_API_KEYS", "")
GROQ_API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]
GROQ_API_KEY: str = GROQ_API_KEYS[0] if GROQ_API_KEYS else os.environ.get("GROQ_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# Context budget (how much retrieved text fits in the prompt)
# This is NOT the same as LLM_MAX_TOKENS (which controls output length).
# Context budget > output budget because the prompt includes multiple chunks + attribution.
CONTEXT_BUDGET: int = 2000
