"""Tests for scripts/health_check.py — semantic K8s health probe."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.health_check import check_health, run_checks


class TestRunChecks:
    @patch("scripts.health_check._check_neo4j", return_value=None)
    @patch("scripts.health_check._check_embedder", return_value=None)
    @patch("scripts.health_check._check_chromadb", return_value=None)
    def test_healthy_when_all_pass(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert errors == []

    @patch("scripts.health_check._check_neo4j", return_value=None)
    @patch("scripts.health_check._check_embedder", return_value=None)
    @patch("scripts.health_check._check_chromadb", return_value="ChromaDB: empty collection")
    def test_unhealthy_when_chromadb_empty(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert len(errors) == 1
        assert "ChromaDB: empty collection" in errors[0]

    @patch("scripts.health_check._check_neo4j", return_value=None)
    @patch("scripts.health_check._check_embedder", return_value=None)
    @patch(
        "scripts.health_check._check_chromadb",
        side_effect=RuntimeError("connection failed"),
    )
    def test_unhealthy_when_chromadb_raises(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert len(errors) == 1
        assert "ChromaDB:" in errors[0]
        assert "connection failed" in errors[0]

    @patch("scripts.health_check._check_neo4j", return_value=None)
    @patch(
        "scripts.health_check._check_embedder",
        return_value="Embedder: wrong dims 100",
    )
    @patch("scripts.health_check._check_chromadb", return_value=None)
    def test_unhealthy_when_embedder_wrong_dims(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert len(errors) == 1
        assert "wrong dims" in errors[0]

    @patch("scripts.health_check._check_neo4j", return_value=None)
    @patch(
        "scripts.health_check._check_embedder",
        side_effect=RuntimeError("model not found"),
    )
    @patch("scripts.health_check._check_chromadb", return_value=None)
    def test_unhealthy_when_embedder_raises(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert len(errors) == 1
        assert "Embedder:" in errors[0]

    @patch(
        "scripts.health_check._check_neo4j",
        side_effect=ConnectionError("neo4j down"),
    )
    @patch("scripts.health_check._check_embedder", return_value=None)
    @patch("scripts.health_check._check_chromadb", return_value=None)
    def test_neo4j_failure_is_non_fatal(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert errors == []  # still healthy — graceful degradation

    @patch("scripts.health_check._check_neo4j", return_value=None)
    @patch(
        "scripts.health_check._check_embedder",
        side_effect=RuntimeError("embedder broken"),
    )
    @patch(
        "scripts.health_check._check_chromadb",
        return_value="ChromaDB: empty collection",
    )
    def test_multiple_errors_collected(self, mock_chroma, mock_emb, mock_neo4j):
        errors = run_checks()
        assert len(errors) == 2
        assert any("ChromaDB" in e for e in errors)
        assert any("Embedder" in e for e in errors)

    def test_skip_checks_via_flags(self):
        # All checks disabled — no errors possible
        errors = run_checks(check_chromadb=False, check_embedder=False, check_neo4j=False)
        assert errors == []


class TestCheckHealth:
    @patch("scripts.health_check.run_checks", return_value=[])
    def test_exits_0_when_healthy(self, mock_run):
        with pytest.raises(SystemExit) as exc_info:
            check_health()
        assert exc_info.value.code == 0

    @patch("scripts.health_check.run_checks", return_value=["ChromaDB: empty"])
    def test_exits_1_when_unhealthy(self, mock_run):
        with pytest.raises(SystemExit) as exc_info:
            check_health()
        assert exc_info.value.code == 1
