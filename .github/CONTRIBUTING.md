# Contributing — RAG Dojo

## Branch Naming

Toda branch segue o padrão: `{tipo}/{spiral}/{descrição-curta}`

**Tipos permitidos:**

| Tipo | Uso |
|------|-----|
| `feat/` | Nova funcionalidade |
| `fix/` | Correção de bug |
| `chore/` | Infra, tooling, CI |
| `analysis/` | Análises exploratórias |
| `docs/` | Documentação |
| `refactor/` | Refatoração sem mudança de comportamento |
| `test/` | Testes novos ou correções de testes |

**Exemplos:**

```
feat/spiral-01/retriever-tests
fix/spiral-01/config-alignment
chore/ci-pipeline-hardening
analysis/score-100-gaps
```

Quando a alteração não é específica de uma spiral, omitir o segmento: `chore/ci-pipeline-hardening`

## Conventional Commits

Todos os commits seguem o padrão: `tipo(scope): mensagem`

```
feat(spiral-01): add chunker with configurable overlap
fix(spiral-01): align CHROMA_COLLECTION_NAME with config
chore(ci): unify pytest runs into single coverage pass
docs(spiral-01): fix strategy names in chunker docstring
```

**Tipos válidos:** feat, fix, chore, docs, refactor, test, ci, perf, analysis

**Scope:** geralmente `spiral-XX` ou `ci`, `docs`, etc.

A CI valida conventional commits automaticamente em PRs.

## Pull Requests

### Regras

1. **Toda alteração entra via PR** — push direto no `main` é bloqueado
2. **1 approval obrigatório** — mantenedor (@HuggsF)
3. **CI deve passar** — test + lint + commit-lint
4. **Coverage 90% por arquivo** — sem exceções

### Antes de abrir PR

Rodar validação local completa:

```bash
make check
```

Isso executa: lint + testes + coverage + validação per-file.

## Cobertura de Testes

- **Regra:** 90% mínimo por arquivo (não global)
- **Enforcement:** `scripts/check_coverage.py --min-percent 90`
- **Sem exceções:** todo arquivo novo precisa de testes que cubram 90%+

## Versionamento

Segue Semantic Versioning alinhado com Spirals:

```
v{major}.{minor}.{patch}

major = número da spiral (1 = Spiral 01, 2 = Spiral 02, ...)
minor = incremento dentro da spiral
patch = fixes
```

| Exemplo | Significado |
|---------|-------------|
| `v0.1.0` | Trabalho inicial, nenhuma spiral completa |
| `v1.0.0` | Spiral 01 completa (aprovada e mergeada) |
| `v1.1.0` | Melhorias na Spiral 01 |
| `v2.0.0` | Spiral 02 completa |

**Importante:** a versão só sobe para `vX.0.0` quando a spiral correspondente está **completa, revisada e mergeada no main**. Trabalho em andamento usa `v0.X.Y`.

## Segurança

Regras invioláveis — ver CLAUDE.md:

1. APENAS dados públicos open-source
2. ZERO chunks sensíveis
3. ZERO cyphers proprietários
4. ZERO modelos de negócio
5. ZERO prompt injection tests
6. ZERO acesso ao sec-forge
