"""Custom exception hierarchy for RAG Dojo.

All exceptions inherit from RAGDojoError so callers can catch the
base class when they don't care about the specific failure type.

Analogy for a C dev:
    errno.h defines EINVAL, ENOMEM, etc. — specific codes under a
    common error domain. RAGDojoError is that domain.
"""

from __future__ import annotations


class RAGDojoError(Exception):
    """Base exception for all RAG Dojo errors."""


class RetrievalError(RAGDojoError):
    """Raised when vector store retrieval fails."""


class GenerationError(RAGDojoError):
    """Raised when LLM generation fails (API error, timeout, etc.)."""


class EmbeddingError(RAGDojoError):
    """Raised when embedding computation fails."""


class ChunkingError(RAGDojoError):
    """Raised when document chunking fails (unreadable file, bad strategy, etc.)."""
