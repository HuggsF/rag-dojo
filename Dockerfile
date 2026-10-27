# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Application code
COPY spiral_01_chunks/ spiral_01_chunks/
COPY spiral_02_graph/ spiral_02_graph/
COPY spiral_03_orchestration/ spiral_03_orchestration/
COPY spiral_04_cognitive/ spiral_04_cognitive/
COPY spiral_05_economics/ spiral_05_economics/
COPY spiral_06_state_of_art/ spiral_06_state_of_art/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pipeline_factory.py contracts.py cli.py Makefile pyproject.toml CLAUDE.md .env.example ./

ENV PYTHONPATH=/app

# Data directory (mount PVC here for persistence)
RUN mkdir -p /app/data

# Keep container alive
CMD ["sleep", "infinity"]
