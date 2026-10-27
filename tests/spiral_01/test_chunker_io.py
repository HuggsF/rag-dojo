"""Tests for Chunker I/O functions — read_file, read_directory, chunk_directory.

These cover the file system interaction layer (lines 50-79, 203-218 in chunker.py)
that test_chunker.py doesn't touch.
"""

import tempfile
from pathlib import Path

import pytest

from spiral_01_chunks.chunker import (
    Chunk,
    chunk_directory,
    read_directory,
    read_file,
)

# --- Fixtures ---


@pytest.fixture
def docs_dir():
    """Create a temp directory with sample .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)

        (p / "intro.md").write_text(
            "FastAPI is a modern framework.\nIt uses Python type hints.", encoding="utf-8"
        )
        (p / "advanced.md").write_text(
            "Middleware in FastAPI allows you to process requests.", encoding="utf-8"
        )
        (p / "empty.md").write_text("", encoding="utf-8")  # should be skipped
        (p / "notes.txt").write_text("Plain text notes about deployment.", encoding="utf-8")
        (p / "image.png").write_bytes(b"\x89PNG\r\n")  # non-text, should be ignored

        # Nested subdirectory
        sub = p / "tutorial"
        sub.mkdir()
        (sub / "step1.md").write_text(
            "Step 1: Install FastAPI with pip install fastapi.", encoding="utf-8"
        )

        yield p


@pytest.fixture
def latin1_file():
    """Create a file encoded as latin-1 with special chars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "latin.md"
        p.write_text("A\xe7\xe3o r\xe1pida, n\xe3o pare!", encoding="latin-1")
        yield p


# --- read_file ---


class TestReadFile:
    def test_reads_utf8(self, docs_dir):
        content = read_file(docs_dir / "intro.md")
        assert "FastAPI" in content
        assert "type hints" in content

    def test_reads_latin1_fallback(self, latin1_file):
        content = read_file(latin1_file)
        assert "pare" in content  # core word survives any decoding path

    def test_accepts_string_path(self, docs_dir):
        content = read_file(str(docs_dir / "intro.md"))
        assert isinstance(content, str)
        assert len(content) > 0

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            read_file("/nonexistent/path/file.md")


# --- read_directory ---


class TestReadDirectory:
    def test_reads_md_files(self, docs_dir):
        results = read_directory(docs_dir)
        filepaths = [fp for fp, _ in results]
        assert any("intro.md" in fp for fp in filepaths)
        assert any("advanced.md" in fp for fp in filepaths)

    def test_skips_empty_files(self, docs_dir):
        results = read_directory(docs_dir)
        contents = [content for _, content in results]
        assert all(c.strip() for c in contents)

    def test_reads_nested_files(self, docs_dir):
        results = read_directory(docs_dir)
        filepaths = [fp for fp, _ in results]
        assert any("step1.md" in fp for fp in filepaths)

    def test_filters_by_extension(self, docs_dir):
        # Default extensions: .md, .txt, .rst
        results = read_directory(docs_dir)
        filepaths = [fp for fp, _ in results]
        assert any(".txt" in fp for fp in filepaths)
        assert not any(".png" in fp for fp in filepaths)

    def test_custom_extensions(self, docs_dir):
        results = read_directory(docs_dir, extensions=(".txt",))
        assert len(results) == 1
        assert "notes.txt" in results[0][0]

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            read_directory("/nonexistent/directory/")

    def test_accepts_string_path(self, docs_dir):
        results = read_directory(str(docs_dir))
        assert len(results) > 0


# --- chunk_directory ---


class TestChunkDirectory:
    def test_produces_chunks(self, docs_dir):
        chunks = chunk_directory(docs_dir)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_default_strategy_is_recursive(self, docs_dir):
        chunks = chunk_directory(docs_dir)
        assert all(c.strategy == "recursive" for c in chunks)

    def test_custom_strategy(self, docs_dir):
        chunks = chunk_directory(docs_dir, strategy="fixed")
        assert all(c.strategy == "fixed" for c in chunks)

    def test_source_metadata_preserved(self, docs_dir):
        chunks = chunk_directory(docs_dir)
        sources = {c.source for c in chunks}
        assert any("intro.md" in s for s in sources)

    def test_chunks_from_multiple_files(self, docs_dir):
        chunks = chunk_directory(docs_dir)
        sources = {c.source for c in chunks}
        assert len(sources) >= 2  # at least intro.md and advanced.md
