"""Tests for Spiral 01 — download_docs module.

All HTTP calls are mocked. Tests validate:
- download_docs() creates output directory
- Downloads files that don't exist locally
- Skips files that already exist
- Handles HTTP errors gracefully
- Uses correct URLs from FASTAPI_DOCS registry
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spiral_01_chunks.download_docs import (
    FASTAPI_DOCS,
    download_docs,
)

# --- Fixtures ---


@pytest.fixture
def output_dir():
    """Create a clean temp directory as output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_mock_response(text="# Mock Content\n\nThis is mock documentation.", status_code=200):
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_error_response():
    """Build a mock httpx response that raises on raise_for_status."""
    import httpx

    resp = MagicMock()
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=MagicMock(), response=MagicMock()
    )
    return resp


# --- FASTAPI_DOCS registry ---


class TestFastapiDocsRegistry:
    def test_has_entries(self):
        assert len(FASTAPI_DOCS) > 0

    def test_keys_are_relative_paths(self):
        for key in FASTAPI_DOCS:
            assert "/" in key or key.endswith(".md")

    def test_values_are_github_urls(self):
        for url in FASTAPI_DOCS.values():
            assert url.startswith("https://raw.githubusercontent.com/fastapi/fastapi/")


# --- download_docs ---


class TestDownloadDocs:
    @patch("spiral_01_chunks.download_docs.OUTPUT_DIR")
    @patch("spiral_01_chunks.download_docs.httpx.Client")
    def test_creates_output_directory(self, MockClient, mock_output_dir, output_dir):
        mock_output_dir.__class__ = Path
        mock_output_dir.mkdir = MagicMock()
        mock_output_dir.rglob = MagicMock(return_value=[])

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _make_mock_response()
        MockClient.return_value = mock_client

        # Patch filepath operations
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "write_text"):
                with patch.object(
                    Path, "parent", new_callable=lambda: property(lambda self: MagicMock())
                ):
                    download_docs()

        mock_output_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("spiral_01_chunks.download_docs.OUTPUT_DIR")
    @patch("spiral_01_chunks.download_docs.httpx.Client")
    def test_skips_existing_files(self, MockClient, mock_output_dir, output_dir, capsys):
        mock_output_dir.__truediv__ = lambda self, x: output_dir / x
        mock_output_dir.mkdir = MagicMock()
        mock_output_dir.rglob = MagicMock(return_value=[output_dir / "test.md"])

        # Create one pre-existing file
        first_key = list(FASTAPI_DOCS.keys())[0]
        existing = output_dir / first_key
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("existing content", encoding="utf-8")

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _make_mock_response()
        MockClient.return_value = mock_client

        download_docs()

        captured = capsys.readouterr()
        assert "[skip]" in captured.out

    @patch("spiral_01_chunks.download_docs.OUTPUT_DIR")
    @patch("spiral_01_chunks.download_docs.httpx.Client")
    def test_downloads_new_files(self, MockClient, mock_output_dir, output_dir, capsys):
        mock_output_dir.__truediv__ = lambda self, x: output_dir / x
        mock_output_dir.mkdir = MagicMock()
        mock_output_dir.rglob = MagicMock(return_value=list(output_dir.rglob("*.md")))

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _make_mock_response()
        MockClient.return_value = mock_client

        download_docs()

        captured = capsys.readouterr()
        assert "[ok]" in captured.out

    @patch("spiral_01_chunks.download_docs.OUTPUT_DIR")
    @patch("spiral_01_chunks.download_docs.httpx.Client")
    def test_handles_http_errors(self, MockClient, mock_output_dir, output_dir, capsys):
        mock_output_dir.__truediv__ = lambda self, x: output_dir / x
        mock_output_dir.mkdir = MagicMock()
        mock_output_dir.rglob = MagicMock(return_value=[])

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _make_mock_error_response()
        MockClient.return_value = mock_client

        # Should not raise — handles errors gracefully
        download_docs()

        captured = capsys.readouterr()
        assert "[fail]" in captured.out

    def test_prints_total_count(self, capsys):
        with patch("spiral_01_chunks.download_docs.OUTPUT_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock()
            mock_dir.rglob = MagicMock(return_value=[Path("a.md"), Path("b.md")])

            mock_filepath = MagicMock()
            mock_filepath.exists.return_value = True
            mock_filepath.parent = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=mock_filepath)

            with patch("spiral_01_chunks.download_docs.httpx.Client") as MockClient:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                MockClient.return_value = mock_client

                download_docs()

            captured = capsys.readouterr()
            assert "Done." in captured.out
