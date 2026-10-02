# Roadmap — Ideas a futuro

Lista viva de desarrollos posibles sobre el buscador semántico, ordenada por
área. Cada idea indica ficheros implicados y esfuerzo aproximado
(S = horas, M = días, L = semanas). Nada aquí es un compromiso: es el
aparcamiento de ideas para cuando apetezca picar algo.

## Cómo usar este documento

1. Elige una idea, crea su rama y sus tests en `tests/`.
2. Si la idea cambia comportamiento, actualiza `README.md` y la doc de `docs/`.
3. Al terminar, marca la casilla `[x]` con la fecha.

## Búsqueda y relevancia

- [ ] **Evaluación con queries fijas** (`tests/`, script nuevo en `scripts/`).
  Set de ~20 queries con resultados esperados que mida precisión/recall.
  Es la idea con más retorno: permite saber si cualquier otro cambio
  (chunking, modelo) mejora o empeora. Esfuerzo: M.
- [ ] **Búsqueda híbrida real** (`src/search.py`, `src/store.py`).
  Hoy `--keyword` es un `$contains` literal. Combinar score vectorial con
  coincidencia de términos (BM25 o ponderación simple) para queries con
  nombres propios o códigos. Esfuerzo: M.
- [ ] **Re-rankeo** (`src/search.py`).
  Segunda pasada sobre los top-N con un cross-encoder o con el propio
  Ollama para reordenar. Esfuerzo: M.
- [ ] **Filtro por fecha** (`src/search.py`, `src/store.py`).
  El metadato `created` ya se indexa pero no se puede filtrar por él
  (`--after` / `--before`). Esfuerzo: S.

## Indexación y contenido

- [ ] **Nuevos formatos** (`src/indexer.py`, parser nuevo).
  PDF, TXT o HTML además de Markdown. El pipeline ya está desacoplado por
  pasos, así que es añadir una etapa de extracción de texto. Esfuerzo: M.
- [ ] **Ajuste de chunking para tu vault** (`src/config.py`, `src/chunker.py`).
  Experimentar con `MAX_CHUNK_TOKENS` / overlap sobre notas reales y medir
  con la evaluación de arriba. Esfuerzo: S.
- [ ] **Más metadatos útiles** (`src/frontmatter_parser.py`, `src/models.py`).
  Extraer enlaces entre notas, conteo de palabras o idioma para filtrar
  y ordenar. Esfuerzo: S–M.

## CLI y uso diario

- [ ] **`watch` como daemon** (`src/cli.py`).
  Hoy es un `run` con debounce fijo. Opciones: más patrones de ignorado,
  reindexado en segundo plano, o servicio systemd. Esfuerzo: S–M.
- [ ] **Nuevos comandos**: `prune` (chunks huérfanos), `duplicates`
  (notas casi idénticas), estadísticas por tag/tipo (`src/cli.py`,
  `src/store.py`). Esfuerzo: S cada uno.
- [ ] **Salida en Markdown** (`src/search.py`).
  Exportar resultados a un `.md` para pegarlos en la vault. Esfuerzo: S.

## Infraestructura y modelos

- [ ] **Probar otros modelos** (solo `EMBEDDING_MODEL` + `index --mode full`).
  `mxbai-embed-large` (más calidad, más lento) o `all-minilm` (más rápido,
  peor en español). El trabajo real es compararlos con tu vault usando la
  evaluación. Esfuerzo: S.
- [ ] **Interfaz web mínima** (módulo nuevo, reutiliza `SearchEngine`).
  FastAPI o Streamlit sobre la misma lógica, sin tocar el pipeline.
  Esfuerzo: M–L.
- [ ] **Migración de vector DB** (`src/store.py`).
  Si el corpus crece (>10K docs), evaluar Qdrant o pgvector. La capa
  `VectorStore` aísla el cambio. Esfuerzo: L.

## Deuda técnica conocida

- [ ] Regenerar el `.venv` con `uv` en la máquina actual (el existente
  apunta a otra ruta y no funciona aquí).
- [ ] Actualizar el año de los headers `Copyright (C)` si el proyecto
  sigue activo en años siguientes.
