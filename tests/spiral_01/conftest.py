"""Shared fixtures for Spiral 01 tests."""

import tempfile

import pytest

from spiral_01_chunks.chunker import Chunk
from spiral_01_chunks.embedder import Embedder
from spiral_01_chunks.retriever import Retriever
from spiral_01_chunks.store import VectorStore


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """4 standard test chunks: FastAPI, Django, Flask, ML."""
    return [
        Chunk(
            text="FastAPI is a modern web framework for building APIs with Python based on type hints.",
            source="fastapi.md",
            chunk_index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Django is a high-level Python web framework that encourages rapid development.",
            source="django.md",
            chunk_index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Flask is a lightweight WSGI web application framework in Python.",
            source="flask.md",
            chunk_index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Machine learning is a subset of artificial intelligence focused on learning from data.",
            source="ml.md",
            chunk_index=0,
            strategy="recursive",
        ),
    ]


@pytest.fixture
def temp_store():
    """Empty vector store in temporary directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(
            persist_dir=tmpdir,
            collection_name="test_collection",
        )
        yield store


@pytest.fixture
def loaded_store(sample_chunks):
    """Vector store pre-loaded with sample chunks + Docker chunk."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(persist_dir=tmpdir, collection_name="test_retriever")
        all_chunks = sample_chunks + [
            Chunk(
                text="Docker containers package applications with their dependencies for consistent deployment.",
                source="docker.md",
                chunk_index=0,
                strategy="recursive",
            ),
        ]
        store.add_documents(all_chunks)
        yield store


@pytest.fixture
def retriever(loaded_store):
    """Retriever backed by loaded store."""
    return Retriever(store=loaded_store)


@pytest.fixture
def embedder():
    """Embedder instance (model loads once, cached for all tests)."""
    return Embedder()
