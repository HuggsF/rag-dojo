# RAG Dojo

**RAG Spiral Training** — do zero ao estado da arte em 6 espirais.

## Stack

| Tech | Uso |
|------|-----|
| Python 3.12 | Runtime |
| ChromaDB | Vector store |
| sentence-transformers | Embeddings (all-MiniLM-L6-v2) |
| Neo4j | Graph store (Spiral 2+) |
| Groq | LLM provider (free tier) |
| rich | CLI output formatting |
| pytest | Testes |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/HuggsF/rag-dojo.git
cd rag-dojo

# 2. Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY (free: https://console.groq.com)

# 5. Download public docs for training data
python spiral_01_chunks/download_docs.py

# 6. Run Spiral 1 pipeline
python -m spiral_01_chunks.pipeline ingest ./data/fastapi_docs/
python -m spiral_01_chunks.pipeline ask "Como criar um middleware no FastAPI?"

# 7. Run tests
pytest
```

## Espirais

| # | Nome | O que constroi | Status |
|---|------|---------------|--------|
| 1 | Chunks | RAG minimo: chunk + embed + retrieve + generate | Active |
| 2 | Graph | GraphRAG hibrido: vetor + Neo4j | Locked |
| 3 | Tensao | Campo de tensao + fallback cascade | Locked |
| 4 | Cognitivo | Bot com memoria + L0/L1/L2 | Locked |
| 5 | Economia | Pipeline L0-L3 + budget tracking | Locked |
| 6 | Estado da Arte | Loop PROPOSE-SOLVE-VERIFY completo | Locked |

## Dados

**APENAS documentacao publica open-source.**
Zero dados sensiveis, zero chunks proprietarios, zero cyphers de negocio.

## Testes

```bash
pytest                          # todos
pytest tests/spiral_01/         # apenas spiral 1
pytest --cov=spiral_01_chunks   # com coverage
```
