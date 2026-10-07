# Task 10: Evaluación fija de recall (legumbres / ZFS)

## Estado: 🟢 BASELINE MEDIDO (2026-10-07)

Baseline contra índice real (994 notas, 6700 chunks, `nomic-embed-text`):

```
query                         recall@5  recall@20 hit  top1
legumbres                     0.50      0.50      1    0.63
plato con garbanzos           0.667     0.667     1    0.76
lentejas                      0.667     1.00      1    0.68
gestión de discos             0.00      0.25      1    0.73
ZFS pool dataset              0.667     1.00      1    0.78
snapshots y replicación ZFS   0.00      0.50      1    0.83
MEDIA recall@5=0.42 hit=1.00 (6 casos)
miss [legumbres]: receta--falafels.md, cocina--hummus.md
miss [gestión de discos]: zfs--guia-basica.md, 260930--zfs-operaciones.md, zfs-almacenamiento.md
```

Lectura: las queries con jerga literal (`ZFS pool dataset`, `plato con
garbanzos`) funcionan; las abstractas (`legumbres`, `gestión de discos`)
fallan. Es el gap hiperónimo/paráfrasis que atacan task-11 y task-12.

Cómo re-correr (la imagen `app` no incluye `scripts/`, hay que montarlo):

```bash
docker compose run --rm -v ./scripts:/scripts:ro -e PYTHONPATH=/app \
  --entrypoint python app /scripts/eval_recall.py --queries /scripts/eval_queries.yml
```

## Objetivo

Dejar de optimizar a ciegas. Crear un set fijo de ~12-20 queries con
resultados esperados que mida `recall@5 / recall@20` antes y después de
cualquier cambio (chunking, enriquecimiento, sinónimos, modelo).

Casos piloto que motivan esta tarea:

* `legumbres` debería traer `garbanzos, lentejas, judías...` (hiperónimo → hipónimo).
* `gestión de discos` debería traer `ZFS, pool, dataset, zpool...` (paráfrasis de dominio).

Hoy el pipeline (`src/search.py` + `src/indexer.py`) hace solo
`query → 1 vector → top-k`. Sin evaluación no se puede saber si un cambio
ayuda o mete ruido.

## Ficheros implicados

| Fichero | Descripción |
|---------|-------------|
| `scripts/eval_queries.yml` | Set de queries + ficheros esperados (aportado por el usuario) |
| `scripts/eval_recall.py` | Corre `SearchEngine.search()` y reporta recall@k |
| `tests/test_eval_recall.py` | Tests del cálculo de recall y parsing del YAML (sin Ollama/Chroma) |
| `docs/learning-path.md` | Cómo usar esta evaluación en la semana 2 |

## Detalle de implementación

### 1. Formato de `eval_queries.yml`

```yaml
# Cada caso: query en lenguaje natural + lista de ficheros que DEBERÍAN salir.
# `source` es el `source_path` indexado (ej: `cocido-garbanzos.md`).
- query: legumbres
  n: 20
  esperados:
    - cocido-garbanzos.md
    - lentejas-estofadas.md

- query: gestión de discos
  n: 20
  esperados:
    - zfs-pools.md
    - smart-monitoreo.md

- query: plato con garbanzos
  n: 20
  esperados:
    - cocido-garbanzos.md
    - hummus.md
```

Reglas:

* `n` es el `n_results` con el que se evalúa (usar 20 para recall, 5 para precisión).
* `esperados` lo rellena el dueño del vault, no se inventa. Si un fichero
  no existe en el índice, el script lo avisa como `missing`.
* Empezar con 6 casos (3 cocina + 3 técnico). Ampliar a 20 con el uso diario.

### 2. Métrica: recall@k

```python
recall@k = len(esperados ∩ top-k sources) / len(esperados)
hit = 1.0 si recall@k > 0 else 0.0  # ¿salió al menos uno?
```

El script reporta por query y media:

```
query                    recall@5  recall@20  hit@20
legumbres                0.50      1.00       1
gestión de discos        0.00      0.50       1
MEDIA                    0.25      0.75       1.0
```

También lista `missing` (esperados que ni siquiera están indexados) y
`scores` del top-1 para diagnosticar (¿0.45 = modelo no conecta? ¿0.70 = solo falta top-k?).

### 3. Uso manual previo (sin código, recomendado)

Antes de automatizar, hacer a mano para entender el problema:

```bash
docker compose run --rm app search "legumbres" -n 20 --explain
docker compose run --rm app search "garbanzos" -n 20 --explain
docker compose run --rm app search "gestión de discos" -n 20 --explain
docker compose run --rm app search "ZFS pool dataset" -n 20 --explain
```

Comparar `Score / Distancia coseno`. Anotar en la tabla de la semana 1 de
`docs/learning-path.md`.

### 4. `scripts/eval_recall.py` — comportamiento

* Lee el YAML, crea `OllamaEmbedder + VectorStore + SearchEngine` con la
  misma `_create_dependencies()` de `src/cli.py`.
* Para cada caso llama `search(query, n_results=n)` sin filtros.
* Compara `r.source` (solo nombre de fichero) contra `esperados`.
* Salida texto para humano + `--json` para comparar entre runs.
* Exit code 0 siempre (es informe, no puerta de CI), salvo error de conexión.

No toca `src/`. Solo lee el índice existente.

### Comentarios didácticos a incluir

* **Por qué recall@20 y no solo @5:** con `DEFAULT_N_RESULTS=5`
  (`src/config.py`) muchos fallos son solo de `top-k`. Si `recall@20=1.0`
  pero `recall@5=0.0`, el problema es ranking/profundidad, no cobertura.
* **Hiperónimo vs paráfrasis:** `legumbres→garbanzo` y
  `gestión de discos→ZFS` son el mismo fenómeno (gap semántico) en dos
  dominios. El modelo `nomic-embed-text` (~274MB) es bueno en similitud
  distributiva (`garbanzo≈lenteja`) pero flojo en taxonomía
  (`legumbre⊃garbanzo`) si la palabra no aparece en el chunk.
* **Tags fuera del vector:** hoy `src/indexer.py` embebe solo
  `título + chunk`. Los tags se guardan en `src/store.py` como metadata
  filtrable, no semántica. La task-11 atacará eso; esta task solo lo mide.

### Tests

* `test_recall_at_k_basico`: esperados `[a.md, b.md]`, top `[a.md, x.md]` → 0.5.
* `test_recall_hit`: ningún esperado en top → hit 0.
* `test_parse_yaml`: carga el esqueleto `eval_queries.yml` sin explotar.
* `test_missing`: esperado que no está en el índice se reporta, no rompe el cálculo.

### Criterio de hecho

* [ ] `scripts/eval_queries.yml` con ≥6 casos reales del usuario.
* [ ] `scripts/eval_recall.py --json` corre contra Chroma real y da tabla.
* [ ] Baseline anotado (fecha + modelo + recall medio) antes de task-11.
