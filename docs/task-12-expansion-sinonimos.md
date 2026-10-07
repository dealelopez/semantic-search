# Task 12: Expansión de query por sinónimos (legumbres / ZFS)

## Estado: 🟡 DISEÑO (listo para implementar tras task-11)

## Por qué (dato real de tu vault, 2026-10-07)

Task-11 no basta por sí sola:

* `receta--falafels.md` (1176 bytes): tags `[recetas]`, cuerpo dice
  `Garbanzos` pero nunca `legumbres`. Enriquecer con tags no crea el puente
  `legumbre ⊃ garbanzo`.
* `receta--alubias-pintas-estofadas.md` (250 bytes): sin frontmatter, pero
  dice literalmente `legumbres secas` → por eso sí sale (0.632).
* `260930--zfs-operaciones.md`: sin frontmatter, jerga `zpool/dataset`
  vs query `gestión de discos` → rank 54+, score 0.64 vs 0.73 de guías genéricas.

El modelo `nomic-embed-text` acerca `garbanzo≈lenteja` pero no
`legumbres→garbanzo` cuando la palabra falta en el chunk. Hay que darle
el puente explícito sin cambiar de modelo.

## Objetivo

Diccionario propio versionado + expansión determinista en
`SearchEngine.search()`, visible en `--explain` y desactivable.

## Ficheros a tocar

| Fichero | Cambio |
|---------|--------|
| `src/synonyms.yml` (nuevo) | Categorías del usuario |
| `src/search.py` | `expand_query()` + fusión multi-query por max-score |
| `src/cli.py` | Flag `--no-expand`, mostrar query expandida en `--explain` |
| `src/config.py` | `SYNONYMS_FILE`, `QUERY_EXPANSION` (default true) |
| `tests/test_search.py` | Tests de expansión y fusión |

## Diseño

### 1. `src/synonyms.yml` (aportado por el usuario, este es el seed)

```yaml
# Clave = término categoría que el usuario buscaría.
# Valores = hipónimos / jerga tal como aparecen en las notas.
legumbres: [garbanzo, garbanzos, lenteja, lentejas, judía, judías, alubia, alubias, haba, habas, guisante, guisantes, soja, falafel, hummus]
zfs: [pool, pools, dataset, datasets, snapshot, snapshots, zpool, zfs, scrub, compresión lz4]
discos: [disco, discos, partición, formatear, smart, sata, ntfs, dd, rsync]
```

Reglas: minúsculas, sin duplicar lo que ya dice la nota, máximo ~15 por clave.
Tú amplías cocina y técnica tras el seed.

### 2. Estrategia: multi-query con fusión (no concatenación ciega)

Concatenar `legumbres (garbanzos, lentejas...)` en un solo embedding diluye.
Mejor:

```
queries = [original] + [1 query por cada 3-4 sinónimos, o 1 query "garbanzos lentejas judías alubias"]
results = [search(q, n) for q in queries]
fusionar por source+heading con max-score, ordenar desc, cortar a n_results
```

* Solo se activa si la query normalizada coincide con una clave (match exacto
  o `in`: `plato con garbanzos` no expande, `legumbres` sí).
* Coste: 2-3 embeddings + queries Chroma por búsqueda con expansión.
  Aceptable en CLI/REPL, documentarlo.

### 3. UX

* `search "legumbres" --explain` muestra `Query expandida: legumbres + [garbanzos, lentejas, ...] (fusión max-score, 2 subqueries)`.
* `--no-expand` y `/no-expand` en REPL para comparar.
* Sin reindex: funciona sobre el índice actual.

## Validación (con task-10)

* `legumbres` debe traer `receta--falafels.md` y `cocina--hummus.md` en top-20
  (hoy fuera del top-100).
* `gestión de discos` debe subir `zfs-*.md` del rank 54+ al top-20 sin echar
  a los genéricos de discos (son relevantes, solo deben convivir).
* Si mete ruido (ej: `legwork` por `legumbres`), recortar el diccionario,
  no el código.

## Tests

* `test_expand_match_exacto`: `legumbres` → 2 subqueries; `plato con garbanzos` → 1.
* `test_fusion_max_score`: mismo chunk en 2 subqueries → se queda el max score, sin duplicados.
* `test_no_expand_flag`: `min_score`/flag desactiva y solo hay 1 llamada a `store.search`.
* `test_synonyms_yml_carga`: claves del seed cargan sin explotar con fichero ausente (fallback: sin expansión).

## Criterio de hecho

* [ ] `synonyms.yml` seed + carga tolerante a ausente.
* [ ] `--explain` muestra expansión.
* [ ] Eval task-10 mejora en los 6 casos sin regresión en el resto.
* [ ] Documentado en `README.md` (sección búsqueda) cómo añadir categorías.
