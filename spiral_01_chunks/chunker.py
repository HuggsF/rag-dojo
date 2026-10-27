"""Chunker — breaks documents into semantic blocks.

Analogy for a C dev:
- A document is a contiguous memory region
- A chunk is a fixed-size block within that region
- Overlap is like having shared bytes between blocks (context preservation)
- If you chunk wrong, you lose context at boundaries (like reading a struct
  that spans two pages without overlap)

Three strategies implemented:
1. fixed     — split by character count (simplest, like fixed-size malloc)
2. recursive — split by paragraph > sentence > word boundaries (production default)
3. sentence  — split by sentence endings (best for Q&A retrieval)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from spiral_01_chunks.config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    source: str  # file path
    chunk_index: int  # position in document
    strategy: str  # which splitter created this
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count (1 token ~ 4 chars for English)."""
        return len(self.text) // 4

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.chunk_index}, tokens~{self.token_estimate}, src={self.source}, '{preview}...')"


def read_file(path: str | Path) -> str:
    """Read a text file, handling common encodings."""
    path = Path(path)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not decode {path} with any known encoding")


def read_directory(
    path: str | Path, extensions: tuple[str, ...] = (".md", ".txt", ".rst")
) -> list[tuple[str, str]]:
    """Read all text files in a directory.

    Returns list of (filepath, content) tuples.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    results = []
    for ext in extensions:
        for file_path in sorted(path.rglob(f"*{ext}")):
            try:
                content = read_file(file_path)
                if content.strip():
                    results.append((str(file_path), content))
            except (ValueError, OSError) as e:
                print(f"Warning: skipping {file_path}: {e}")
    return results


def chunk_fixed(
    text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Fixed-size chunking — like malloc with fixed block size.

    Simple, predictable, but may split mid-sentence.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    splits = splitter.split_text(text)
    return [
        Chunk(text=s, source=source, chunk_index=i, strategy="fixed") for i, s in enumerate(splits)
    ]


def chunk_recursive(
    text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Recursive chunking — tries multiple separators in order.

    Like a smart allocator that respects paragraph > sentence > word boundaries.
    This is the production default.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    splits = splitter.split_text(text)
    return [
        Chunk(text=s, source=source, chunk_index=i, strategy="recursive")
        for i, s in enumerate(splits)
    ]


def chunk_sentence(
    text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Sentence-based chunking — respects sentence boundaries.

    Splits on sentence endings (. ! ?) then groups into blocks.
    Better for Q&A because answers rarely span sentence boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[Chunk] = []
    current_chunk: list[str] = []
    current_length = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current_length + len(sentence) > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_idx,
                    strategy="sentence",
                )
            )
            chunk_idx += 1

            # Keep overlap sentences
            overlap_chars = 0
            overlap_start = len(current_chunk)
            for j in range(len(current_chunk) - 1, -1, -1):
                overlap_chars += len(current_chunk[j])
                if overlap_chars >= overlap:
                    overlap_start = j
                    break
            current_chunk = current_chunk[overlap_start:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(
            Chunk(
                text=" ".join(current_chunk),
                source=source,
                chunk_index=chunk_idx,
                strategy="sentence",
            )
        )

    return chunks


# Strategy registry
STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence,
}


def chunk_text(
    text: str,
    source: str,
    strategy: str = "recursive",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk text using the specified strategy.

    Args:
        text: Raw text to chunk.
        source: Source file path (for metadata).
        strategy: One of 'fixed', 'recursive', 'sentence'.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap between chunks in characters.

    Returns:
        List of Chunk objects.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy](text, source, chunk_size, overlap)


def chunk_directory(
    path: str | Path,
    strategy: str = "recursive",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk all documents in a directory.

    Returns all chunks with source metadata preserved.
    """
    documents = read_directory(path)
    all_chunks: list[Chunk] = []
    for filepath, content in documents:
        chunks = chunk_text(content, filepath, strategy, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks
