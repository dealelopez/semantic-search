# Ruta de aprendizaje — de búsqueda literal a búsqueda por conceptos

Proyecto para aprender búsqueda semántica usando tu propio vault.
Hilo conductor: dos casos del mismo fenómeno (gap semántico):

* Cocina: `legumbres` → `garbanzo, lenteja, judía...`
* Técnico: `gestión de discos` → `ZFS, pool, dataset, zpool...`

Cada semana tiene objetivo, comandos, ficheros a leer y auto-check.
Tiempo estimado: 2-3h/semana durante 4 semanas.

## Semana 1 — Diagnosticar (sin cambiar código)

**Objetivo:** entender por qué falla y cuantificarlo a mano.

Leer:

* `src/search.py` (método `search`, líneas ~67-122)
* `src/indexer.py` (qué texto se embebe, líneas ~141-145)
* `src/store.py` (qué se guarda como metadata vs vector)

Hacer:

```bash
docker compose run --rm app status
docker compose run --rm app search "legumbres" -n 20 --explain
docker compose run --rm app search "garbanzos" -n 20 --explain
docker compose run --rm app search "gestión de discos" -n 20 --explain
docker compose run --rm app search "ZFS pool dataset" -n 20 --explain
# Variantes:
docker compose run --rm app search "legumbres" -n 20 --unique
docker compose run --rm app search "legumbres" -n 20 --json | head -50
```

Anotar tabla:

| query | top-1 score | ¿sale lo esperado? | hipótesis |
|-------|-------------|--------------------|-----------|
| legumbres | ? | | ¿falta la palabra en el chunk? |
| gestión de discos | ? | | ¿jerga distinta? |

Auto-check:

* [ ] Sé explicar qué es un embedding y qué es distancia coseno sin mirar `README.md`.
* [ ] Sé decir qué parte del texto de una nota entra al vector y qué parte solo es metadata.
* [ ] Tengo 3 ejemplos query → fichero esperado por dominio (para la semana 2).

## Semana 2 — Medir (task-10)

**Objetivo:** montar evaluación fija para no optimizar a ciegas.

Leer: `docs/task-10-evaluacion-recall.md`.

Hacer:

1. Rellenar `scripts/eval_queries.yml` con tus 6 casos (3 cocina + 3 técnico).
2. Correr el esqueleto:
   ```bash
   python3 scripts/eval_recall.py --queries scripts/eval_queries.yml
   python3 scripts/eval_recall.py --queries scripts/eval_queries.yml --json
   ```
   (Necesita ChromaDB + Ollama arrancados; solo lee, no reindexa.)
3. Anotar baseline: fecha + `EMBEDDING_MODEL` + recall medio @5/@20.

Ficheros a crear/tocar: solo `scripts/`, ningún `src/`.

Auto-check:

* [ ] Sé calcular `recall@k` a mano con 1 ejemplo.
* [ ] Sé distinguir un fallo de cobertura (`recall@20=0`) de uno de ranking (`recall@20=1, recall@5=0`).

Siguiente (previsto, aún no implementar):

* `docs/task-11-enriquecimiento-embeddings.md` — embeder `título+tags+tipo+heading+chunk`.
* `docs/task-12-expansion-sinonimos.md` — `src/synonyms.yml` propio + multi-query.

## Semana 3 — Enriquecer + expandir (con reindex)

**Objetivo:** mejorar recall con tus datos, midiendo con la eval de S2.

Prerrequisito: baseline de S2 guardado.

Pasos previstos (no ejecutar aún):

1. Task-11: cambiar texto embebido en `src/indexer.py` para incluir tags/tipo.
   Requiere `index --mode full`. Comparar recall antes/después.
2. Task-12: crear `src/synonyms.yml`:
   ```yaml
   legumbres: [garbanzo, lenteja, judía, alubia, haba, guisante]
   zfs: [pool, dataset, snapshot, zpool]
   ```
   Expandir query en `src/search.py` o lanzar multi-query con fusión max-score.
3. Re-correr `eval_recall.py` tras cada cambio. Quedarse solo con lo que suba recall sin hundir precisión.

Auto-check:

* [ ] Sé explicar por qué hay que hacer `full` tras cambiar lo embebido.
* [ ] Sé desactivar la expansión (`--no-expand`) y ver su efecto en `--explain`.

## Semana 4 — Híbrida, rerank y modelos (plan B)

**Objetivo:** decidir con datos si compensa subir complejidad.

Leer: `docs/roadmap.md` (secciones búsqueda híbrida, re-rankeo, otros modelos).

Experimentos previstos:

1. Híbrida: combinar score vectorial + `where_document $contains` (`--keyword` actual en `src/search.py:97`).
2. Re-rank top-20 con cross-encoder u Ollama.
3. Cambiar `EMBEDDING_MODEL` a `mxbai-embed-large` + `full reindex`, comparar con misma eval.

Auto-check:

* [ ] Sé explicar trade-off latencia/calidad de cada técnica.
* [ ] Tengo una tabla comparativa modelo A vs B con mi vault, no con benchmarks genéricos.

## Qué me falta aportarte (para seguir)

1. Lista inicial de categorías: ¿qué incluye `legumbres` para ti? ¿Qué otras categorías de cocina y de técnica quieres?
2. 3 pares query → fichero esperado por dominio (los de la tabla S1).
3. Confirmar si el formato `eval_queries.yml` te encaja o prefieres JSON.

Con (1) y (2) se rellena el YAML de ejemplo y se cierra la task-10.
