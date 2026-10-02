# Task 3: Chunker híbrido de Markdown

## Estado: ✅ HECHO

## Objetivo

Implementar `chunker.py` con la estrategia híbrida de chunking (por headings → por párrafos → por tamaño fijo), diseñado para ser extensible. Cada chunk incluye el texto, la ruta de headings (para mostrar en resultados), y su posición en el documento.

## Ficheros a crear/modificar

| Fichero | Acción | Descripción |
|---------|--------|-------------|
| `src/models.py` | MODIFICAR | Añadir dataclass `Chunk` |
| `src/chunker.py` | CREAR | Clase abstracta `ChunkingStrategy` + `HybridChunker` |
| `tests/fixtures/nota_larga_con_secciones.md` | CREAR | Nota con h2/h3, alguna sección >300 tokens |
| `tests/fixtures/nota_corta_sin_headings.md` | CREAR | Nota sin headings, solo párrafos |
| `tests/test_chunker.py` | CREAR | Tests unitarios del chunker |

## Detalle de implementación

### src/models.py — Añadir Chunk

```python
@dataclass
class Chunk:
    text: str               # Contenido textual del chunk (lo que se embede)
    heading_path: str       # Ruta de headings (ej: "## Config > ### Redis")
    chunk_index: int        # Posición del chunk dentro de la nota (0, 1, 2...)
    token_count: int        # Tokens estimados del chunk
    source_path: str        # Ruta del fichero origen
```

### src/chunker.py — Estrategia híbrida

**Clase abstracta (para extensibilidad futura):**

```python
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, body: str, metadata: NoteMetadata) -> list[Chunk]:
        ...
```

Esto permite crear otras estrategias después (ej: `FixedSizeChunker`, `ParagraphChunker`, `SemanticChunker`) sin cambiar el resto del código.

**Clase HybridChunker(ChunkingStrategy):**

Algoritmo principal:

```
1. Parsear el markdown buscando líneas que empiecen con #, ##, ###
2. Dividir en secciones por heading

3. ¿Tiene headings?
   SÍ → Para cada sección:
        ¿Sección > MAX_CHUNK_TOKENS (300)?
            SÍ → Subdividir por párrafos (líneas en blanco)
                  ¿Párrafo > MAX_CHUNK_TOKENS?
                      SÍ → Corte por tamaño fijo con overlap de OVERLAP_TOKENS (50)
                      NO → Chunk directo
            NO → Chunk entero de la sección

   NO → Dividir por párrafos (líneas en blanco)
        Agrupar párrafos consecutivos hasta alcanzar MIN_CHUNK_TOKENS (100)
        ¿Grupo > MAX_CHUNK_TOKENS?
            SÍ → Corte por tamaño fijo con overlap
            NO → Chunk directo
```

**Funciones auxiliares:**

- `estimate_tokens(text: str) -> int`: Aproximación rápida del número de tokens.
  - Fórmula: `len(text.split()) * 1.3` (para español, que tiene palabras más largas que inglés).
  - Comentario explicando por qué no usamos un tokenizer real (como tiktoken): para chunking, una aproximación del ±20% es suficiente, y evitamos añadir una dependencia pesada. La precisión exacta importa más cuando se paga por token (APIs de OpenAI), no cuando controlamos el modelo localmente.

- `_split_by_headings(body: str) -> list[tuple[str, str]]`: Devuelve lista de `(heading, content)`. El heading incluye el nivel (`## Título`). El content es todo el texto hasta el siguiente heading.

- `_split_by_paragraphs(text: str) -> list[str]`: Divide por líneas en blanco (`\n\n`). Filtra párrafos vacíos.

- `_split_by_size(text: str, max_tokens: int, overlap_tokens: int) -> list[str]`: Corte mecánico por tamaño fijo con solapamiento.

- `_group_small_paragraphs(paragraphs: list[str], min_tokens: int) -> list[str]`: Agrupa párrafos consecutivos hasta alcanzar el mínimo.

**heading_path:**

Cada chunk registra la "ruta" de headings bajo la que cae. Ejemplo:

```markdown
## Configuración
### Redis
Texto sobre Redis...
```

El chunk con "Texto sobre Redis..." tendría `heading_path = "## Configuración > ### Redis"`. Esto se usa para mostrar en los resultados de búsqueda dónde está el fragmento dentro del documento.

### Comentarios didácticos a incluir

- Por qué el chunking es probablemente la decisión más importante en un sistema de búsqueda semántica
- Qué pasa cuando el chunk es muy grande (embedding difuso, representa demasiados conceptos) vs muy pequeño (pierde contexto, "Esto funciona bien" no tiene suficiente info)
- Qué es el overlap y por qué evita perder ideas que caen en el borde entre dos chunks
- Por qué la estrategia híbrida es más pragmática que una estrategia pura (headings pueden ser genéricos como "Paso 3", párrafos pueden ser muy cortos o muy largos)
- Por qué la estimación de tokens es suficiente para chunking (no necesitamos precisión de facturación)
- Por qué usamos una clase abstracta (principio Open/Closed: abierto a extensión, cerrado a modificación)

### Fixtures de test

**nota_larga_con_secciones.md:**

Nota con frontmatter, título h1, dos secciones h2 (una corta <300 tokens, una larga >300 tokens con subsecciones h3). Esto ejercita:
- Split por headings
- Subdivisión de sección larga
- Heading path multinivel

**nota_corta_sin_headings.md:**

Nota con frontmatter pero sin headings en el body. Solo párrafos separados por líneas en blanco. Algunos párrafos muy cortos (<50 tokens). Esto ejercita:
- Fallback a split por párrafos
- Agrupación de párrafos pequeños

### Tests

- `test_chunk_por_headings`: nota con secciones h2 → genera un chunk por sección, heading_path correcto
- `test_subdivision_seccion_larga`: sección >300 tokens → se subdivide, cada sub-chunk ≤ MAX_CHUNK_TOKENS
- `test_agrupacion_parrafos_cortos`: nota sin headings con párrafos cortos → se agrupan hasta MIN_CHUNK_TOKENS
- `test_heading_path_multinivel`: nota con h2 + h3 → heading_path = "## X > ### Y"
- `test_chunk_index_secuencial`: chunk_index va 0, 1, 2... para cada nota
- `test_estimate_tokens`: verificar que la estimación es razonable (±30% respecto a un conteo manual)
- `test_nota_vacia`: body vacío → devuelve lista vacía (no explota)
- `test_overlap_en_corte_por_tamano`: chunk grande cortado por tamaño → overlap presente entre consecutivos
