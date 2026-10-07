# Task 11: Enriquecimiento del texto embebido (tags + título + heading)

## Estado: 🟡 DISEÑO (listo para implementar)

## Por qué (dato real de tu vault, 2026-10-07)

* `search "legumbres" -n 100`: solo 2 recetas en top-100
  (`receta--alubias-pintas-estofadas.md` 0.632,
  `receta--berza-judias-verdes.md` 0.616). `receta--falafels.md`
  (garbanzos) **no aparece ni en top-100**, aunque `search "garbanzos"`
  la trae a 0.711.
* `search "gestión de discos"`: genéricos de discos a 0.73;
  `zfs-*.md` enterrados en rank 54-69 a ~0.64.
* Causa en código: `src/indexer.py:145` y `:294` embeben solo
  `f"{title}\n\n{c.text}"`. Los tags/tipo/heading se guardan en
  `src/store.py:151-162` como metadata filtrable, **no entran al vector**.
* Límite conocido: muchas notas no tienen tags útiles.
  `receta--falafels.md` tiene `tags: [recetas]` (genérico) y
  `260930--zfs-operaciones.md` no tiene frontmatter.
  Por eso esta task **ayuda pero no basta** — necesita task-12 (sinónimos).

## Objetivo

Incluir contexto estructurado en el texto que se embebe, sin cambiar lo que
se guarda como `document` en Chroma (el preview sigue siendo el original).

## Ficheros a tocar

| Fichero | Cambio |
|---------|--------|
| `src/indexer.py` | Nueva función `build_enriched_text(metadata, chunk)` usada en `full_reindex` e `_process_note` |
| `src/config.py` | Flags `EMBED_ENRICH_TAGS`, `EMBED_ENRICH_HEADING` (default true) para A/B |
| `tests/test_indexer.py` | Tests del formato enriquecido + que el document guardado no cambia |
| `example.env` | Documentar las 2 flags nuevas |

## Diseño

```python
def build_enriched_text(metadata: NoteMetadata, chunk_text: str, heading: str) -> str:
    header_parts = [metadata.title]
    if config.EMBED_ENRICH_TAGS and metadata.tags:
        header_parts.append(f"[{' | '.join(metadata.tags)}]")
    if config.EMBED_ENRICH_HEADING and heading:
        header_parts.append(heading)
    return "\n".join(header_parts) + "\n\n" + chunk_text
```

Ejemplo:

```
antes: "Falafels y salsa de tahini - Receta\n\n- Garbanzos secos..."
después: "Falafels y salsa de tahini - Receta\n[recetas]\n## Ingredientes\n\n- Garbanzos secos..."
```

* Si la nota es `sin-tipo` y sin tags, el formato degrada al actual (sin corchetes vacíos).
* No se toca `_deterministic_id` ni el esquema de metadata: el `document`
  en `upsert_chunks` sigue siendo `chunk.text` original.

## Reindex requerido

Sí: `index --mode full`. Los vectores viejos (sin contexto) no son
comparables con queries nuevas. Avisar en CLI/report.

## Validación

1. Anotar baseline task-10 antes.
2. `index --mode full`, re-correr `scripts/eval_recall.py`.
3. Éxito si `recall@20` sube en `gestión de discos` (tags `tecnologia/homelab`
   ayudan) sin bajar en `legumbres`. Si `legumbres→falafels` sigue a 0,
   es lo esperado: lo resuelve task-12.

## Tests

* `test_enriched_incluye_tags_y_heading`: metadata con tags + heading → el texto contiene ambos.
* `test_enriched_sin_tags_degrada`: sin tags → sin corchetes, igual que antes.
* `test_document_guardado_es_original`: `upsert_chunks` recibe `chunk.text` sin prefijo.
* `test_flag_desactiva_enriquecimiento`: con flags a false, `build_enriched_text` devuelve `title + texto`.

## Criterio de hecho

* [ ] Flags en `config.py` + `example.env`.
* [ ] `full_reindex` e `incremental` usan la misma función.
* [ ] Tests en verde + eval task-10 re-medida y anotada.
