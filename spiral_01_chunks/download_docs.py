"""Download public documentation for training data.

Downloads FastAPI docs from GitHub (public, MIT licensed).
These are the ONLY data sources allowed — see CLAUDE.md security rules.
"""

from __future__ import annotations

from pathlib import Path

import httpx

# Public FastAPI docs (MIT license — https://github.com/fastapi/fastapi)
FASTAPI_DOCS = {
    "tutorial/first-steps.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/first-steps.md",
    "tutorial/path-params.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/path-params.md",
    "tutorial/query-params.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/query-params.md",
    "tutorial/body.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/body.md",
    "tutorial/middleware.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/middleware.md",
    "tutorial/dependencies.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
    "tutorial/security.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/security/index.md",
    "tutorial/cors.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/cors.md",
    "tutorial/sql-databases.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/sql-databases.md",
    "tutorial/background-tasks.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/background-tasks.md",
    "tutorial/testing.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/testing.md",
    "tutorial/debugging.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/debugging.md",
    "advanced/middleware.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/advanced/middleware.md",
    "advanced/events.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/advanced/events.md",
    "advanced/websockets.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/advanced/websockets.md",
    "advanced/testing-dependencies.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/advanced/testing-dependencies.md",
    "deployment/concepts.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/deployment/concepts.md",
    "deployment/docker.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/deployment/docker.md",
}

OUTPUT_DIR = Path("data/fastapi_docs")


def download_docs() -> None:
    """Download all FastAPI docs to local directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(FASTAPI_DOCS)} FastAPI doc files...")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for filename, url in FASTAPI_DOCS.items():
            filepath = OUTPUT_DIR / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if filepath.exists():
                print(f"  [skip] {filename} (already exists)")
                continue

            try:
                response = client.get(url)
                response.raise_for_status()
                filepath.write_text(response.text, encoding="utf-8")
                print(f"  [ok]   {filename} ({len(response.text)} chars)")
            except httpx.HTTPError as e:
                print(f"  [fail] {filename}: {e}")

    total = sum(1 for _ in OUTPUT_DIR.rglob("*.md"))
    print(f"\nDone. {total} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    download_docs()
