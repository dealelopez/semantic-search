# Task 2: Parser de frontmatter YAML

## Estado: ✅ HECHO

## Objetivo

Implementar `frontmatter_parser.py` que lee un archivo MD, extrae el frontmatter YAML como metadata estructurada (`NoteMetadata`) y devuelve el cuerpo markdown limpio. Manejar el caso de notas sin frontmatter (el ~2% del corpus).

## Ficheros a crear

| Fichero | Descripción |
|---------|-------------|
| `src/models.py` | Dataclasses `NoteMetadata` y `ParsedNote` |
| `src/frontmatter_parser.py` | Función `parse_note(filepath) -> ParsedNote` |
| `tests/fixtures/nota_con_frontmatter.md` | Fixture: nota completa con frontmatter v2 |
| `tests/fixtures/nota_sin_frontmatter.md` | Fixture: nota sin bloque YAML |
| `tests/test_frontmatter.py` | Tests unitarios del parser |

## Detalle de implementación

### src/models.py — Dataclasses

```python
@dataclass
class NoteMetadata:
    source_path: str        # Ruta relativa del fichero (ej: "proyecto-x.md")
    title: str              # Título legible (de frontmatter o nombre de fichero)
    tags: list[str]         # Lista de tags (ej: ["nota", "ia", "estado/seed"])
    note_type: str          # Primer tag = tipo de nota (ej: "nota", "proyecto", "receta")
    created: str | None     # Fecha de creación (YYYY-MM-DD) o None
    updated: str | None     # Última modificación o None
    extra_fields: dict      # Campos adicionales del frontmatter (status, maker, etc.)

@dataclass
class ParsedNote:
    metadata: NoteMetadata  # Metadatos extraídos del frontmatter
    body: str               # Cuerpo markdown limpio (sin frontmatter, sin wikilinks)
```

### src/frontmatter_parser.py — Lógica

Función principal: `parse_note(filepath: Path) -> ParsedNote`

1. Leer el fichero con `python-frontmatter`:
   - Si tiene bloque `---` al inicio → extrae YAML + body
   - Si no tiene frontmatter → body = todo el fichero

2. Construir `NoteMetadata`:
   - `title`: campo `title` del YAML, o nombre del fichero sin extensión si no hay
   - `tags`: campo `tags` del YAML, o lista vacía
   - `note_type`: primer elemento de `tags` (según sistema v2 del usuario), o `"sin-tipo"` si no hay tags
   - `created` / `updated`: campos del YAML, convertidos a string
   - `extra_fields`: todos los demás campos del YAML que no sean los core (title, tags, created, updated)

3. Limpiar el body:
   - Eliminar wikilinks de Obsidian: `[[texto]]` → `texto`, `[[target|display]]` → `display`
   - Usar regex: `r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]'` → `\1`
   - Motivo: los wikilinks añaden ruido a los embeddings. `[[RAG — Retrieval-Augmented Generation]]` no aporta significado semántico, pero `RAG — Retrieval-Augmented Generation` sí.

4. Devolver `ParsedNote(metadata, body)`

Función auxiliar: `scan_notes(directory: Path) -> list[Path]`
- Escanea recursivamente `directory` buscando `*.md`
- Ignora directorios cuyos nombres estén en `config.NOTES_IGNORE_PATTERNS` (ej: `.obsidian`, `.trash`, `.git`, `_templates`)
- Devuelve lista ordenada de paths

### Comentarios didácticos a incluir

- Qué es frontmatter y por qué se usa en sistemas de notas (Obsidian, Jekyll, Hugo)
- Por qué el frontmatter es valioso como metadata filtrable en un buscador semántico
- Por qué limpiamos wikilinks antes de generar embeddings (ruido vs señal)
- Cómo el tipo de nota (primer tag) permite filtrar resultados por categoría

### Fixtures de test

**nota_con_frontmatter.md:**
```yaml
---
created: 2026-09-15
updated: 2026-09-20
title: Embeddings y búsqueda semántica
tags:
  - nota
  - inteligencia-artificial
  - embeddings
  - estado/seedling
---

# Embeddings y búsqueda semántica

Los embeddings son vectores numéricos que representan el significado
de un texto. Ver [[RAG — Retrieval-Augmented Generation]] para más info.

## Modelos recomendados

| Modelo | Tamaño | Calidad |
|--------|--------|---------|
| nomic-embed-text | 274MB | Muy buena |
| all-minilm | 45MB | Buena |
```

**nota_sin_frontmatter.md:**
```markdown
# Nota suelta sin metadatos

Esta nota no tiene bloque YAML al inicio.
Es simplemente markdown plano.

## Sección

Algo de contenido aquí con un [[enlace wiki]].
```

### Tests

- `test_parse_con_frontmatter`: parsear fixture con frontmatter → verificar que `metadata.title == "Embeddings y búsqueda semántica"`, `metadata.tags == ["nota", "inteligencia-artificial", ...]`, `metadata.note_type == "nota"`, `metadata.created == "2026-09-15"`
- `test_parse_sin_frontmatter`: parsear fixture sin frontmatter → `metadata.title` = nombre del fichero, `metadata.tags == []`, `metadata.note_type == "sin-tipo"`, body = todo el contenido
- `test_wikilinks_limpiados`: el body de `nota_con_frontmatter.md` debe contener `RAG — Retrieval-Augmented Generation` pero NO `[[RAG — Retrieval-Augmented Generation]]`
- `test_body_no_contiene_yaml`: el body no debe empezar con `---`
- `test_scan_notes`: escanear directorio de fixtures → devuelve lista de paths `.md`
