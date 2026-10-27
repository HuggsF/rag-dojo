# CLAUDE.md — RAG Dojo

## Projeto

- **Nome**: RAG Dojo
- **Descricao**: RAG Spiral Training — do zero ao estado da arte em 6 espirais
- **Repo**: `HuggsF/rag-dojo`

## Tech Stack

| Tech | Uso |
|------|-----|
| Python 3.12 | Runtime |
| ChromaDB | Vector store (embedding persistence) |
| sentence-transformers | Embeddings (all-MiniLM-L6-v2, 384 dims) |
| Neo4j | Graph store (Spiral 2+) |
| Groq | LLM provider (free tier — llama-3.3-70b) |
| Ollama | LLM local fallback (mistral:7b — sem rate limit) |
| pytest | Testes |
| rich | CLI output formatting |

## Regras de Seguranca (INVIOLAVEIS)

1. **APENAS dados publicos open-source** — FastAPI docs, LangChain docs, Neo4j docs, Python stdlib
2. **ZERO chunks sensiveis** de qualquer projeto privado
3. **ZERO cyphers proprietarios** — apenas Cypher queries sobre dados publicos
4. **ZERO modelos de negocio** — sem business logic de clientes
5. **ZERO prompt injection tests** — sem payloads de seguranca
6. **ZERO acesso ao sec-forge** — repo separado, sem cross-reference

## Estrutura de Espirais

Cada espiral segue o padrao: Doc (B) + Dojo (A) + Missao (C)

| Espiral | Diretorio | Dependencia |
|---------|-----------|-------------|
| 1 — Chunks | `spiral_01_chunks/` | Nenhuma |
| 2 — Graph | `spiral_02_graph/` | Espiral 1 (ChromaDB) |
| 3 — Tensao | `spiral_03_orchestration/` | Espiral 2 (hybrid retriever) |
| 4 — Cognitivo | `spiral_04_cognitive/` | Espirais 1-3 (sistema pra monitorar) |
| 5 — Economia | `spiral_05_economics/` | Espirais 1-4 (custos pra rastrear) |
| 6 — Estado da Arte | `spiral_06_state_of_art/` | Todas (integracao completa) |

## Infraestrutura (SSOT)

O ambiente canonico de execucao e **Kubernetes**. O Windows local serve apenas para
desenvolvimento de codigo e testes unitarios (com mocks).

### Pods (namespace: default)

| Pod | Funcao | Dados |
|-----|--------|-------|
| `rag-test` | Executa pipeline RAG (ingestao, queries) | ChromaDB embutido (`./data/chroma_db`) |
| `neo4j` | Graph database | Grafo de conhecimento (entities + relations) |

### Servico Neo4j

| De onde | URI | Quando usar |
|---------|-----|-------------|
| Pod `rag-test` (K8s) | `bolt://neo4j:7687` | **SEMPRE** — e o default do codigo |
| Windows local | `bolt://localhost:7687` | Apenas via port-forward para debug |

Para acessar Neo4j localmente: `kubectl port-forward svc/neo4j 7687:7687`

### Regra do .env

O `.env` vive **DENTRO do pod `rag-test`**, NAO no repositorio local.
O repo tem apenas `.env.example` como referencia. **Nunca criar `.env` no repo.**

### Onde rodar os pipelines

| Comando | Onde rodar |
|---------|-----------|
| Ingestao (chunks, graph) | Dentro do pod `rag-test` via `kubectl exec` |
| Queries (ask, ask-hybrid) | Dentro do pod `rag-test` via `kubectl exec` |
| Testes unitarios | Windows local com `py -3.12 -m pytest` (mocks) |
| Lint | Windows local com `py -3.12 -m ruff check .` |

### Validacao do ambiente

```bash
py -3.12 scripts/check_env.py          # Valida tudo (K8s + local)
py -3.12 scripts/check_env.py --local  # Pula checks K8s (CI)
```

## Docker

### Build da imagem

```bash
# SEMPRE buildar no context `default` — e o que o K8s Docker Desktop enxerga
docker --context default build -t rag-dojo:local -f Dockerfile .

# Verificar tamanho (esperado ~3-4GB, NAO 12GB+)
docker images rag-dojo:local --format "{{.Size}}"
```

**IMPORTANTE:**
- O `.dockerignore` DEVE existir — sem ele a imagem fica com 12GB+ (inclui .git, data, caches)
- SEMPRE usar `--context default` no build — o K8s Docker Desktop nao enxerga imagens do context `desktop-linux`
- Apos rebuild, deletar e recriar o pod para usar a nova imagem

### Deploy dos pods

```bash
# Criar PVC (uma vez)
kubectl apply -f k8s/pvc-rag-data.yaml

# Criar/recriar pods
kubectl delete pod rag-test -n default --ignore-not-found
kubectl apply -f k8s/pod-rag-test.yaml
kubectl apply -f k8s/pod-neo4j.yaml
```

### Limpeza Docker (quando necessario)

```bash
docker system prune -f          # Remove containers/images/cache orfaos
docker volume prune -f          # Remove volumes sem container
docker builder prune -f         # Limpa build cache
```

## Como Rodar

### Setup local (dev/testes)

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pytest -v --tb=short
py -3.12 -m ruff check .
```

### Execucao real (dentro do pod rag-test)

```bash
kubectl exec rag-test -n default -- python spiral_01_chunks/download_docs.py
kubectl exec rag-test -n default -- python -m spiral_01_chunks.pipeline ingest ./data/fastapi_docs/
kubectl exec rag-test -n default -- python -m spiral_02_graph.pipeline ingest-graph ./data/fastapi_docs/
kubectl exec rag-test -n default -- python -m spiral_02_graph.pipeline ask "Como criar um middleware no FastAPI?"
```

### Fallback Ollama (quando Groq rate limited)

```bash
# Requer Ollama rodando no Windows host com mistral:7b
kubectl exec rag-test -n default -- python scripts/k8s_ingest_ollama.py
```

## LLM Providers

| Provider | Modelo | Rate Limit | Uso |
|----------|--------|------------|-----|
| Groq (primary) | llama-3.3-70b-versatile | 100k TPD free tier | Entity extraction, generation |
| Ollama (fallback) | mistral:7b | Sem limite (local) | Entity extraction quando Groq rate limited |

### Ollama Setup (fallback local)

```bash
# Instalar: winget install Ollama.Ollama (ou https://ollama.com)
ollama pull mistral:7b

# Usar via Python:
from spiral_02_graph.entity_extractor import OllamaEntityExtractor
extractor = OllamaEntityExtractor(model="mistral:7b")
result = extractor.extract_from_chunk("texto aqui", "chunk_id")
```

Ollama roda em `http://localhost:11434`. Nao requer API key.

## SSOTs

| SSOT | Conteudo |
|------|----------|
| `CLAUDE.md` (este) | Config do projeto |
| `.env.example` | Variaveis obrigatorias |
| `requirements.txt` | Dependencias |

## Conventional Commits

Seguir padrao da org: `feat(spiral-01): add chunker with configurable overlap`
