# Task 6: Indexer — Orquestación de indexación completa e incremental

## Estado: ✅ HECHO

## Objetivo

Implementar `indexer.py` que orquesta el pipeline completo: leer notas → parsear frontmatter → trocear en chunks → generar embeddings → almacenar en ChromaDB. Dos modos: `full` (reindexar todo desde cero) y `incremental` (solo nuevos/modificados/eliminados).

## Ficheros a crear/modificar

| Fichero | Acción | Descripción |
|---------|--------|-------------|
| `src/models.py` | MODIFICAR | Añadir dataclass `IndexReport` |
| `src/indexer.py` | CREAR | Clase `Indexer` con `full_reindex` e `incremental_index` |
| `tests/test_indexer.py` | CREAR | Tests con mocks de embedder y store |

## Detalle de implementación

### src/models.py — Añadir IndexReport

```python
@dataclass
class IndexReport:
    notes_processed: int     # Notas procesadas (nuevas + modificadas)
    notes_skipped: int       # Notas sin cambios (solo incremental)
    notes_deleted: int       # Notas eliminadas del índice (solo incremental)
    chunks_created: int      # Total de chunks generados
    errors: list[str]        # Lista de errores (fichero + mensaje)
    duration_seconds: float  # Tiempo total de ejecución
```

### src/indexer.py — Clase Indexer

**Constructor:**

```python
class Indexer:
    def __init__(
        self,
        embedder: OllamaEmbedder,
        store: VectorStore,
        chunker: ChunkingStrategy,
    ):
        # Recibe las dependencias inyectadas.
        # No crea nada por sí mismo — solo orquesta.
```

**Método: full_reindex(notes_dir: str) -> IndexReport**

Reindexación completa: borra todo y reprocesa desde cero.

```
1. store.reset_collection()  → borra la colección entera
2. paths = scan_notes(notes_dir)  → lista de todos los .md
3. Para cada nota (con barra de progreso rich):
   a. parse_note(path)  → ParsedNote (metadata + body)
   b. chunker.chunk(body, metadata)  → list[Chunk]
   c. Si no hay chunks (nota vacía), skip
   d. embedder.embed_documents([chunk.text for chunk in chunks])  → list[vector]
   e. store.upsert_chunks(chunks, embeddings, metadata)
   f. Si error en cualquier paso → log del error, continúa con la siguiente nota
4. Devolver IndexReport con totales y duración
```

Cuándo usar:
- Primera indexación (la colección está vacía)
- Después de cambiar el modelo de embeddings (los vectores son incompatibles)
- Después de cambiar la estrategia de chunking (los chunks son diferentes)
- Si sospechas que el índice está corrupto o desincronizado
- Estimación: ~3-8 minutos para 400 notas en CPU

**Método: incremental_index(notes_dir: str) -> IndexReport**

Indexación incremental: solo procesa lo que cambió.

```
1. indexed = store.get_indexed_sources()  → {"nota.md": "hash123", ...}
2. paths = scan_notes(notes_dir)  → lista de .md en disco

3. Para cada .md en disco:
   a. content_hash = compute_content_hash(path)
   b. ¿Está en indexed?
      NO → es NUEVO → procesar (parse → chunk → embed → upsert)
      SÍ → ¿Hash diferente?
           SÍ → es MODIFICADO → store.delete_by_source(path), luego procesar
           NO → SIN CAMBIOS → skip

4. Para cada source en indexed que NO está en disco:
   → es ELIMINADO → store.delete_by_source(source)

5. Devolver IndexReport con desglose
```

Cuándo usar:
- Después de añadir o editar algunas notas
- Como comando habitual del día a día
- Mucho más rápido que full_reindex porque solo procesa lo que cambió
- No es válido si cambiaste el modelo o la estrategia de chunking

**Función auxiliar: compute_content_hash(filepath: Path) -> str**

```python
def compute_content_hash(filepath: Path) -> str:
    """
    Calcula un hash MD5 del contenido del fichero.
    Se usa para detectar si una nota cambió desde la última indexación.
    MD5 es suficiente aquí — no necesitamos seguridad criptográfica,
    solo detectar cambios en el contenido.
    """
    content = filepath.read_bytes()
    return hashlib.md5(content).hexdigest()
```

**Barra de progreso (rich):**

```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
) as progress:
    task = progress.add_task("Indexando notas...", total=len(paths))
    for path in paths:
        # ... procesar nota ...
        progress.advance(task)
```

### Comentarios didácticos a incluir

- **Full vs incremental — el trade-off**: Full reindex es simple y seguro (siempre correcto), pero lento (~3-8 min para 400 notas). Incremental es rápido (segundos si solo cambió una nota), pero más complejo y asume que el modelo y la estrategia de chunking no cambiaron. La mayoría de veces usarás incremental; full se reserva para "reconstruir desde cero".

- **Por qué hash del contenido y no fecha de modificación**: La fecha de modificación (`mtime`) del fichero puede cambiar sin que el contenido cambie (ej: rsync, git checkout, sync de Obsidian). El hash del contenido es la única forma fiable de detectar cambios reales.

- **Idempotencia**: Puedes ejecutar `full_reindex` o `incremental_index` las veces que quieras sin efectos secundarios. Full borra y recrea. Incremental usa upsert (si el ID ya existe, se sobreescribe). No se crean duplicados.

- **Resiliencia a errores**: Si una nota falla (frontmatter malformado, timeout de Ollama), el indexer la salta y continúa con la siguiente. Los errores se acumulan en el report para revisarlos después. No queremos que una nota problemática pare la indexación de las otras 399.

- **Inyección de dependencias**: El Indexer no crea sus dependencias (embedder, store, chunker) — las recibe en el constructor. Esto permite: (1) testear con mocks, (2) cambiar la implementación sin tocar el indexer (ej: cambiar ChromaDB por Qdrant, o Ollama por OpenAI).

### Tests

Todos los tests usan mocks de `embedder` y `store` para no depender de Docker ni Ollama:

```python
@pytest.fixture
def mock_embedder():
    embedder = Mock(spec=OllamaEmbedder)
    # embed_documents devuelve vectores dummy de 768 dims
    embedder.embed_documents.return_value = [[0.1] * 768]
    return embedder

@pytest.fixture
def mock_store():
    store = Mock(spec=VectorStore)
    store.get_indexed_sources.return_value = {}
    return store
```

- `test_full_reindex_procesa_todas`: directorio con 3 notas → report.notes_processed == 3
- `test_full_reindex_resetea_coleccion`: verificar que `store.reset_collection()` se llama antes de procesar
- `test_incremental_detecta_nuevos`: store vacío + 2 notas en disco → 2 procesadas, 0 skipped, 0 deleted
- `test_incremental_detecta_modificados`: store con hash "abc", disco con hash "xyz" → 1 procesada, store.delete_by_source llamado
- `test_incremental_detecta_eliminados`: store con "nota-x.md" pero no está en disco → notes_deleted == 1
- `test_incremental_skip_sin_cambios`: store y disco con mismo hash → notes_skipped == 1
- `test_error_en_nota_no_para_proceso`: mock embedder que lanza excepción para una nota → las demás se procesan, error en report
- `test_nota_vacia_se_salta`: nota con body vacío (sin chunks) → notes_processed incrementa pero chunks_created no
