"""Environment validation script for RAG Dojo.

Checks that the infrastructure is correctly configured before running pipelines.
Validates: Python version, K8s pods, Neo4j connectivity, ChromaDB, and env vars.

Usage:
    python scripts/check_env.py              # Full validation
    python scripts/check_env.py --local      # Skip K8s checks (CI/local dev)

Exit codes:
    0 — all checks pass
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# ARCHITECTURE NOTE
# ---------------------------------------------------------------------------
# The canonical runtime environment is Kubernetes:
#   - Pod `rag-test`:  runs the RAG pipeline (ChromaDB embedded, Python code)
#   - Pod `neo4j`:     graph database (bolt://neo4j:7687 from inside K8s)
#   - Service `neo4j`: ClusterIP exposing bolt:7687 + http:7474
#
# The .env file lives INSIDE the rag-test pod, NOT in the local repository.
# The local repo has only .env.example as reference.
#
# For local development/testing:
#   - Use `kubectl port-forward svc/neo4j 7687:7687` to access Neo4j
#   - Set NEO4J_URI=bolt://localhost:7687 when running locally
#   - ChromaDB data on Windows is for unit tests only (mocked)
#   - All real data lives inside the K8s pods
# ---------------------------------------------------------------------------


def check_python_version() -> bool:
    """Verify Python 3.12.x is being used."""
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if major == 3 and minor == 12:
        print(f"  [+] PASS  Python {version}")
        return True
    print(f"  [!] FAIL  Python {version} (expected 3.12.x)")
    print("           Use: py -3.12 scripts/check_env.py")
    return False


def check_required_packages() -> bool:
    """Verify critical packages are installed."""
    packages = {
        "chromadb": "chromadb",
        "neo4j": "neo4j",
        "sentence_transformers": "sentence-transformers",
        "groq": "groq",
        "rich": "rich",
        "pytest": "pytest",
    }
    all_ok = True
    for import_name, display_name in packages.items():
        try:
            __import__(import_name)
            print(f"  [+] PASS  {display_name} installed")
        except ImportError:
            print(f"  [!] FAIL  {display_name} NOT installed")
            all_ok = False
    return all_ok


def check_env_vars() -> bool:
    """Check that required env vars are set (inside pod) or warn (local)."""
    required = ["GROQ_API_KEY"]
    optional_with_defaults = {
        "NEO4J_URI": "bolt://neo4j:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "(from .env in pod)",
    }

    all_ok = True
    for var in required:
        val = os.environ.get(var, "")
        if val:
            print(f"  [+] PASS  {var} is set")
        else:
            print(f"  [~] WARN  {var} not set (required for LLM generation)")
            # Not a hard fail — tests run without it

    for var, default in optional_with_defaults.items():
        val = os.environ.get(var, "")
        if val:
            print(f"  [+] PASS  {var} = {val}")
        else:
            print(f"  [~] INFO  {var} not set (default: {default})")

    return all_ok


def check_k8s_pods() -> bool:
    """Verify K8s pods are running and neo4j is accessible from rag-test."""
    import subprocess

    all_ok = True

    # Check pods exist
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", "default", "-o", "wide", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

        neo4j_found = False
        rag_test_found = False
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                name, ready, status = parts[0], parts[1], parts[2]
                if "neo4j" in name:
                    neo4j_found = True
                    if status == "Running":
                        print(f"  [+] PASS  Pod {name} is {status} ({ready})")
                    else:
                        print(f"  [!] FAIL  Pod {name} is {status}")
                        all_ok = False
                if "rag-test" in name:
                    rag_test_found = True
                    if status == "Running":
                        print(f"  [+] PASS  Pod {name} is {status} ({ready})")
                    else:
                        print(f"  [!] FAIL  Pod {name} is {status}")
                        all_ok = False

        if not neo4j_found:
            print("  [!] FAIL  Pod neo4j not found in namespace default")
            all_ok = False
        if not rag_test_found:
            print("  [!] FAIL  Pod rag-test not found in namespace default")
            all_ok = False

    except FileNotFoundError:
        print("  [!] FAIL  kubectl not found in PATH")
        all_ok = False
    except subprocess.TimeoutExpired:
        print("  [!] FAIL  kubectl timed out — is the cluster running?")
        all_ok = False

    # Check neo4j service
    try:
        result = subprocess.run(
            ["kubectl", "get", "svc", "neo4j", "-n", "default", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("  [+] PASS  Service neo4j exists")
        else:
            print("  [!] FAIL  Service neo4j not found")
            all_ok = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # Already reported above

    return all_ok


def check_neo4j_connectivity() -> bool:
    """Test Neo4j connectivity from current environment."""
    try:
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")

        if not password:
            print("  [~] SKIP  Neo4j connectivity (NEO4J_PASSWORD not set)")
            return True

        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS cnt")
            count = result.single()["cnt"]
        driver.close()
        print(f"  [+] PASS  Neo4j at {uri} — {count} nodes")
        return True
    except Exception as e:
        print(f"  [!] FAIL  Neo4j at {uri} — {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Dojo environment validator")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Skip K8s checks (for CI or local-only dev)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RAG Dojo — Environment Validation")
    print("=" * 60)

    all_ok = True

    print("\n[Python]")
    all_ok &= check_python_version()

    print("\n[Packages]")
    all_ok &= check_required_packages()

    print("\n[Environment Variables]")
    all_ok &= check_env_vars()

    if not args.local:
        print("\n[Kubernetes]")
        all_ok &= check_k8s_pods()

    print("\n[Neo4j Connectivity]")
    all_ok &= check_neo4j_connectivity()

    print("\n" + "=" * 60)
    if all_ok:
        print("  All checks passed.")
    else:
        print("  Some checks failed — review the output above.")
    print("=" * 60 + "\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
