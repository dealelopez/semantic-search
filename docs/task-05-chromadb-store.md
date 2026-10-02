# Task 5: Store ChromaDB — Operaciones CRUD vectoriales

## Estado: ⬜ PENDIENTE

## Objetivo

Implementar `store.py` que encapsula todas las operaciones con ChromaDB: crear/obtener colección, upsert chunks con embeddings y metadata, buscar por similitud, listar y borrar documentos. Este módulo es la capa de persistencia del sistema.

## Ficheros a crear/modificar

| Fichero | Acción | Descripción |
|---------|--------|-------------|
| `src/models.py` | MODIFICAR | Añadir dataclass `SearchResult` |
| `src/store.py` | CREAR | Clase `VectorStore` con todas las operaciones ChromaDB |
| `tests/test_store.py` | CREAR | Tests con ChromaDB efímero (sin Docker) |

## Detalle de implementación

### src/models.py — Añadir SearchResult

```python
@dataclass
class SearchResult:
    source: str         # Ruta del fichero origen (ej: "proyecto-x.md")
    heading: str        # Heading del chunk (ej: "## Configuración > ### Redis")
    text: str           # Texto completo del chunk
    score: float        # Similitud (0.0 a 1.0, mayor = más similar)
    note_type: str      # Tipo de nota (primer tag)
    tags: list[str]     # Lista de tags
    created: str | None # Fecha de creación
```

### src/store.py — VectorStore

**Constructor:**

```python
class VectorStore:
    def __init__(self, host: str, port: int, collection_name: str):
        # Conecta con chromadb.HttpClient(host=host, port=port)
        # Obtiene o crea la colección con:
        #   get_or_create_collection(
        #       name=collection_name,
        #       metadata={"hnsw:space": "cosine"}
        #   )
```

**Métodos:**

`upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]], note_metadata: NoteMetadata) -> int`

1. Para cada chunk + embedding, prepara:
   - **id**: hash determinista `md5(source_path + ":" + str(chunk_index))`.
     IDs deterministas permiten hacer upsert idempotente: si reindexas el mismo fichero, se sobreescriben los chunks existentes en lugar de duplicarlos.
   - **embedding**: el vector de 768 floats
   - **document**: el texto del chunk (ChromaDB lo almacena para devolverlo en búsquedas)
   - **metadata**: dict con campos filtrables:
     ```python
     {
         "source": "proyecto-x.md",
         "heading": "## Configuración > ### Redis",
         "tags": ["nota", "ia", "embeddings"],  # Array nativo — ChromaDB 1.5+ soporta listas en metadata
         "note_type": "nota",
         "created": "2026-09-15",
         "content_hash": "a1b2c3..."       # Hash del contenido del fichero (para incremental)
     }
     ```
2. Hace `collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)`
3. Devuelve el número de chunks insertados/actualizados.

`search(query_embedding: list[float], n_results: int = 5, where_filter: dict | None = None) -> list[SearchResult]`

1. Ejecuta `collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where_filter)`
2. ChromaDB devuelve:
   ```python
   {
       "ids": [["id1", "id2", ...]],
       "distances": [[0.12, 0.25, ...]],  # Distancia coseno (0 = idéntico)
       "documents": [["texto1", "texto2", ...]],
       "metadatas": [[{...}, {...}, ...]]
   }
   ```
3. Convierte distancia a score de similitud: `score = 1 - distance` (para coseno, distancia va de 0 a 2, pero en la práctica los resultados relevantes están entre 0 y 1)
4. Construye lista de `SearchResult` ordenada por score descendente.
5. Tags ya vienen como lista nativa de ChromaDB (no requiere parseo).

`get_indexed_sources() -> dict[str, str]`

1. Obtiene todos los documentos de la colección: `collection.get(include=["metadatas"])`
2. Agrupa por `source` y extrae el `content_hash`.
3. Devuelve `{"proyecto-x.md": "a1b2c3...", "nota-y.md": "d4e5f6..."}`.
4. Necesario para la indexación incremental (saber qué ya está indexado y si cambió).

`delete_by_source(source_path: str) -> int`

1. Busca todos los IDs con `metadata.source == source_path`.
2. `collection.delete(ids=ids_encontrados)`
3. Devuelve el número de chunks eliminados.
4. Se usa cuando un fichero cambió (se borran los chunks viejos antes de insertar los nuevos) o cuando un fichero fue eliminado del disco.

`collection_stats() -> dict`

1. `collection.count()` → total de chunks.
2. Obtiene metadatas para contar fuentes únicas.
3. Devuelve:
   ```python
   {
       "total_chunks": 1847,
       "total_sources": 312,
       "collection_name": "notas"
   }
   ```

`reset_collection() -> None`

1. `self._client.delete_collection(name)` → borra la colección entera.
2. Recrea la colección vacía con `get_or_create_collection(...)`.
3. Se usa para reindexación completa (empezar de cero).

### Comentarios didácticos a incluir

- **Qué es una colección vectorial**: Equivalente a una "tabla" en bases de datos relacionales, pero optimizada para buscar por similitud entre vectores. Cada entrada tiene un ID, un vector (embedding), un documento (texto), y metadatos (key-value para filtrar).

- **HNSW (Hierarchical Navigable Small World)**: El algoritmo que ChromaDB usa internamente para buscar vecinos cercanos de forma eficiente. Sin HNSW, buscar el vector más parecido entre 10.000 requeriría comparar con todos (O(n)). HNSW lo hace en O(log n) construyendo un grafo navegable por capas. Nosotros no tocamos HNSW directamente, solo le decimos que use distancia coseno.

- **Espacio coseno**: Al crear la colección con `hnsw:space: cosine`, le decimos a ChromaDB que use distancia coseno para comparar vectores. Otros espacios posibles: `l2` (euclídea) e `ip` (producto punto). Para embeddings de texto, coseno es el estándar porque es invariante a la magnitud del vector.

- **IDs deterministas**: Si generamos IDs random, cada reindexación crea nuevas entradas (duplicados). Con IDs deterministas (hash del path + posición del chunk), un upsert sobreescribe el chunk existente. Esto es la base del indexado incremental y la idempotencia.

- **Tags como array nativo**: ChromaDB 1.5+ soporta arrays en metadata (strings, ints, floats, bools). Almacenamos los tags como `["nota", "ia", "embeddings"]` y usamos `$contains` para filtrar. En versiones anteriores (<1.0) no se soportaban listas y había que usar CSV o booleanos individuales.

### Tests (con ChromaDB efímero)

Para tests rápidos sin Docker, usamos `chromadb.EphemeralClient()` que crea una instancia en memoria.

**Dependencia**: `EphemeralClient` está en el paquete `chromadb` completo, no en `chromadb-client` (que es solo el cliente HTTP). Por eso `requirements-dev.txt` incluye `chromadb==1.5.3` además de `chromadb-client==1.5.3` (heredado de `requirements.txt`) — el primero es solo para tests.

```python
@pytest.fixture
def store():
    import chromadb
    client = chromadb.EphemeralClient()
    # Monkey-patch para usar EphemeralClient en lugar de HttpClient
    ...
```

- `test_upsert_y_count`: insertar 3 chunks → `collection_stats()["total_chunks"] == 3`
- `test_upsert_idempotente`: insertar los mismos chunks 2 veces → count sigue siendo 3
- `test_search_devuelve_ordenado`: insertar chunks con vectores conocidos, buscar → resultados ordenados por score
- `test_search_con_filtro_tipo`: insertar chunks de tipo "nota" y "proyecto", filtrar por "nota" → solo devuelve notas
- `test_search_con_filtro_tags`: filtrar por tag "ia" usando `$contains` sobre array → devuelve solo chunks con ese tag
- `test_delete_by_source`: insertar chunks de 2 ficheros, borrar 1 → solo quedan los del otro
- `test_get_indexed_sources`: insertar chunks de 3 ficheros → devuelve dict con 3 entradas y sus hashes
- `test_reset_collection`: insertar chunks, resetear → count == 0
- `test_search_coleccion_vacia`: buscar sin datos → lista vacía (no explota)
