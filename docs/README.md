# Plan de implementación — Buscador Semántico Local

## Resumen

Buscador semántico local para ~200-400 notas en Markdown con frontmatter YAML.
Usa embeddings (Ollama + nomic-embed-text) para buscar por significado, no por palabras exactas.
Todo dockerizado (ChromaDB + app Python). CLI con modos one-shot y REPL interactivo.

## Arquitectura

```
┌──────────────────────────────────────────────┐
│            Docker Compose                     │
│                                               │
│  ┌────────────┐     ┌─────────────────────┐  │
│  │  ChromaDB   │◄────│  App Python (CLI)   │  │
│  │  :8000      │     │                     │  │
│  │  vectores   │     │  parsea → trocea    │  │
│  └────────────┘     │  → embede → busca   │  │
│                      └──────────┬──────────┘  │
│                                 │              │
└─────────────────────────────────┼──────────────┘
                                  │ HTTP
                                  ▼
                         ┌──────────────────┐
                         │  Ollama (remoto)  │
                         │  nomic-embed-text │
                         │  :11434           │
                         └──────────────────┘
```

## Tasks

> Ver también [Roadmap](roadmap.md) con ideas a futuro (búsqueda híbrida,
> evaluación, nuevos formatos, interfaz web...).

| # | Task | Estado | Ficheros principales |
|---|------|--------|---------------------|
| 1 | [Scaffolding](task-01-scaffolding.md) | [OK] HECHO | docker-compose.yml, Dockerfile, config.py |
| 2 | [Frontmatter parser](task-02-frontmatter-parser.md) | ⬜ PENDIENTE | models.py, frontmatter_parser.py |
| 3 | [Chunker híbrido](task-03-chunker.md) | ⬜ PENDIENTE | chunker.py |
| 4 | [Embeddings client](task-04-embeddings-client.md) | ⬜ PENDIENTE | embeddings.py |
| 5 | [ChromaDB store](task-05-chromadb-store.md) | ⬜ PENDIENTE | store.py |
| 6 | [Indexer](task-06-indexer.md) | ⬜ PENDIENTE | indexer.py |
| 7 | [Search engine](task-07-search-engine.md) | ⬜ PENDIENTE | search.py |
| 8 | [CLI](task-08-cli.md) | ⬜ PENDIENTE | cli.py |
| 9 | [README + polish](task-09-readme-polish.md) | ⬜ PENDIENTE | README.md, notas de ejemplo |

## Orden de implementación

Las tasks se ejecutan en orden secuencial (cada una depende de las anteriores):

```
Task 1 (scaffolding)
  └→ Task 2 (frontmatter) → define models.py que usan todos
      └→ Task 3 (chunker) → necesita NoteMetadata de Task 2
          └→ Task 4 (embeddings) → independiente, pero se testea aquí
              └→ Task 5 (store) → necesita Chunk de Task 3
                  ├→ Task 6 (indexer) → orquesta Tasks 2-5
                  └→ Task 7 (search) → usa Tasks 4-5
                      └→ Task 8 (CLI) → ensambla Tasks 6-7
                          └→ Task 9 (README + notas ejemplo)
```

> **Nota:** Task 6 (indexer) y Task 7 (search) son independientes entre sí —
> ambas dependen de Tasks 4-5 pero no se usan mutuamente. Se recomienda
> implementar Task 6 primero porque Task 7 necesita datos indexados para
> probar búsquedas de forma realista.

## Decisiones de diseño

- **ChromaDB sobre Qdrant/pgvector**: para <10K documentos, ChromaDB es más simple (Python-nativa, sin configuración extra). Si el corpus crece, migrar a Qdrant es sencillo.
- **nomic-embed-text sobre alternativas**: mejor equilibrio calidad/velocidad para contenido en español en CPU-only. 768 dims, 274MB, context window 8192 tokens.
- **Chunking híbrido**: combina las ventajas de partir por headings (respeta estructura) con fallback a párrafos/tamaño fijo (robusto con cualquier formato).
- **Indexación dual**: full (simple, segura) + incremental (rápida) — ambas implementadas para aprender las diferencias.
- **CLI con argparse**: sin dependencias externas, suficiente para los subcomandos necesarios.
- **rich para output**: hace que la terminal sea legible con colores, tablas y barras de progreso.
