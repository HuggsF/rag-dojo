.PHONY: help install dev test test-fast test-cov lint format check security clean ingest ask download-docs ingest-graph ask-hybrid graph-stats visualize check-env k8s-ingest k8s-ingest-graph k8s-ask k8s-stats k8s-port-forward

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Setup development environment
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env || true

test: ## Run all tests
	pytest -v --tb=short

test-fast: ## Run tests excluding slow markers
	pytest -v -m "not slow"

test-cov: ## Run tests with coverage + per-file validation (90%/arquivo)
	pytest -v --tb=short --cov --cov-report=term-missing --cov-report=json
	python scripts/check_coverage.py --min-percent 90

lint: ## Run linter (ruff)
	ruff check .
	ruff format --check .

format: ## Auto-format code
	ruff format .

security: ## Run security scan (bandit)
	bandit -r spiral_01_chunks/ spiral_02_graph/ spiral_03_orchestration/ spiral_04_cognitive/ spiral_05_economics/ spiral_06_state_of_art/ -ll

check: lint test-cov security ## Full validation before push (lint + test + coverage 90%/file + security)

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage coverage.json

ingest: ## Ingest FastAPI docs into vector store
	python -m spiral_01_chunks.pipeline ingest ./data/fastapi_docs/

ask: ## Ask a question (usage: make ask Q="your question")
	python -m spiral_01_chunks.pipeline ask "$(Q)"

download-docs: ## Download public docs for training
	python spiral_01_chunks/download_docs.py

# --- Spiral 02: Graph RAG ---

ingest-graph: ## Build knowledge graph from ingested chunks
	python -m spiral_02_graph.pipeline ingest-graph ./data/fastapi_docs/

ask-hybrid: ## Ask using hybrid retrieval (usage: make ask-hybrid Q="your question")
	python -m spiral_02_graph.pipeline ask "$(Q)"

graph-stats: ## Show knowledge graph statistics
	python -m spiral_02_graph.pipeline stats

visualize: ## Export graph visualization to HTML (usage: make visualize [E="entity name"])
	python -m spiral_02_graph.pipeline visualize $(E)

check-env: ## Validate environment (Python, K8s pods, Neo4j connectivity)
	python scripts/check_env.py

# --- Kubernetes: canonical runtime (all data lives here) ---

k8s-ingest: ## Ingest FastAPI docs into ChromaDB inside rag-test pod
	kubectl exec rag-test -n default -- python spiral_01_chunks/download_docs.py
	kubectl exec rag-test -n default -- python -m spiral_01_chunks.pipeline ingest ./data/fastapi_docs/

k8s-ingest-graph: ## Build knowledge graph inside K8s (rag-test → neo4j pod)
	kubectl exec rag-test -n default -- python -m spiral_02_graph.pipeline ingest-graph ./data/fastapi_docs/

k8s-ask: ## Ask using hybrid retrieval inside K8s (usage: make k8s-ask Q="your question")
	kubectl exec rag-test -n default -- python -m spiral_02_graph.pipeline ask "$(Q)"

k8s-stats: ## Show graph stats from K8s neo4j pod
	kubectl exec rag-test -n default -- python -m spiral_02_graph.pipeline stats

k8s-ingest-ollama: ## Build knowledge graph via Ollama fallback (when Groq rate limited)
	kubectl exec rag-test -n default -- python scripts/k8s_ingest_ollama.py

k8s-port-forward: ## Forward neo4j ports to localhost for local debugging
	kubectl port-forward svc/neo4j 7687:7687 7474:7474
