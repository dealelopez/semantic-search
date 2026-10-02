# Task 7: Motor de búsqueda y formateo de resultados

## Estado: ✅ HECHO

## Objetivo

Implementar `search.py` que recibe una query en lenguaje natural, genera el embedding, busca en ChromaDB, y devuelve resultados formateados para terminal. Soporte para filtros por tags y tipo de nota.

## Ficheros a crear

| Fichero | Descripción |
|---------|-------------|
| `src/search.py` | Clase `SearchEngine` con métodos `search` y `format_results` |
| `tests/test_search.py` | Tests unitarios con mocks |

## Detalle de implementación

### src/search.py — SearchEngine

**Constructor:**

```python
class SearchEngine:
    def __init__(self, embedder: OllamaEmbedder, store: VectorStore):
        # Recibe embedder (genera vectores de queries) y store (busca en ChromaDB)
```

**Método: search(query, n_results=5, filter_tags=None, filter_type=None) -> list[SearchResult]**

```
1. Generar embedding de la query:
   query_embedding = embedder.embed_query(query)
   → Internamente añade prefijo "search_query: " al texto

2. Construir filtro ChromaDB (where clause):
   - Si filter_type: where = {"note_type": {"$eq": filter_type}}
   - Si filter_tags: where = {"tags": {"$contains": tag}}
     (ChromaDB 1.5+ soporta $contains sobre arrays en metadata.
      Los tags se almacenan como lista: ["nota", "ia", "embeddings"])
   - Si ambos: where = {"$and": [{tipo}, {tags}]}
   - Si varios tags: where = {"$and": [{"tags": {"$contains": t}} for t in tags]}
   - Si ninguno: where = None

3. Buscar en ChromaDB:
   results = store.search(
       query_embedding=query_embedding,
       n_results=n_results,
       where_filter=where
   )

4. Devolver lista de SearchResult
```

**Método: format_results(query: str, results: list[SearchResult], elapsed: float) -> None**

Formatea e imprime los resultados usando `rich` para output bonito en terminal:

```
Resultados para: "cómo funcionan las redes neuronales"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 1. redes-neuronales-intro.md > ## Qué es una red neuronal
    Similitud: 0.92
    Tags: nota · inteligencia-artificial · ml
    Una red neuronal es un modelo computacional inspirado en el
       cerebro humano que aprende patrones a partir de datos de
       entrenamiento. Cada neurona artificial recibe inputs...

 2. deep-learning-basics.md > ## Capas y neuronas
    Similitud: 0.87
    Tags: nota · deep-learning
    Una red neuronal profunda tiene múltiples capas ocultas
       entre la entrada y la salida. Cada capa transforma los...

 3. tensorflow-tutorial.md
    Similitud: 0.81
    Tags: nota · python · tensorflow
    TensorFlow permite definir y entrenar redes neuronales
       con pocas líneas de código. Primero se define la...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3 resultados en 0.34s
```

Implementación con rich:

```python
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def format_results(self, query: str, results: list[SearchResult], elapsed: float) -> None:
    # Imprime directamente en consola con rich
    console.print(f"\nResultados para: \"{query}\"")
    console.print("━" * 55)

    if not results:
        console.print("[dim]No se encontraron resultados.[/dim]")
        return

    for i, r in enumerate(results, 1):
        # Línea principal: número + fichero + heading
        header = f" {i}. {r.source}"
        if r.heading:
            header += f" > {r.heading}"
        console.print(f"[bold]{header}[/bold]")

        # Score con color según valor
        color = "green" if r.score >= 0.8 else "yellow" if r.score >= 0.6 else "red"
        console.print(f"    Similitud: [{color}]{r.score:.2f}[/{color}]")

        # Tags
        if r.tags:
            tags_str = " · ".join(r.tags)
            console.print(f"    Tags: {tags_str}")

        # Preview del contenido (truncado a PREVIEW_MAX_CHARS)
        preview = r.text[:PREVIEW_MAX_CHARS]
        if len(r.text) > PREVIEW_MAX_CHARS:
            preview += "..."
        # Indentar el preview y reemplazar saltos de línea
        preview_lines = preview.replace("\n", "\n       ")
        console.print(f"    {preview_lines}")
        console.print()  # Línea en blanco entre resultados

    console.print("━" * 55)
    console.print(f"{len(results)} resultado(s) en {elapsed:.2f}s")
```

**Caso especial — sin resultados:**

Si la búsqueda no devuelve resultados, mostrar un mensaje útil:
```
Resultados para: "receta de pizza"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No se encontraron resultados.

Sugerencias:
   - ¿Has indexado las notas? Ejecuta: index --mode full
   - Prueba con otros términos o sinónimos
   - Si usas filtros, prueba sin ellos
```

### Comentarios didácticos a incluir

- **Distancia coseno → score de similitud**: ChromaDB devuelve "distance" (menor = más similar). Nosotros lo convertimos a "score" (mayor = más similar) con `score = 1 - distance`. Es más intuitivo: 0.95 = muy similar, 0.30 = poco relevante.

- **El mismo modelo para indexar y buscar**: Es crítico que el embedding de la query se genere con el mismo modelo que los embeddings de los documentos. Los vectores de modelos diferentes viven en espacios matemáticos distintos y no son comparables. Es como intentar medir distancias mezclando kilómetros y millas.

- **Filtrado por metadata**: ChromaDB primero filtra por metadata (operación rápida sobre un índice) y luego busca por similitud solo entre los resultados filtrados. Esto es mucho más eficiente que buscar entre todos y filtrar después. En ChromaDB 1.5+, los tags se almacenan como arrays nativos y se filtran con `$contains`, sin necesidad de parsear strings CSV.

- **El "semantic gap"**: A veces los resultados de búsqueda semántica sorprenden. Si buscas "vacaciones" podrías encontrar un texto sobre "días de descanso laboral" aunque no use la palabra "vacaciones". Eso es el poder del embedding — captura significado, no palabras. Pero también puede fallar: si el concepto no está bien representado en los datos de entrenamiento del modelo, el matching puede ser impreciso.

### Tests

- `test_search_sin_filtros`: verificar que `store.search` se llama sin `where_filter`
- `test_search_con_filtro_tipo`: pasar `filter_type="proyecto"` → `where_filter` incluye `{"note_type": {"$eq": "proyecto"}}`
- `test_search_con_filtro_tags`: pasar `filter_tags=["ia"]` → `where_filter` incluye `{"tags": {"$contains": "ia"}}`
- `test_search_con_ambos_filtros`: tipo + tags → `where_filter` usa `$and`
- `test_search_con_multiples_tags`: pasar `filter_tags=["ia", "ml"]` → `where_filter` usa `$and` con un `$contains` por tag
- `test_format_resultados_vacio`: lista vacía → muestra mensaje "no se encontraron resultados"
- `test_format_resultados_trunca_preview`: texto >200 chars → preview termina en "..."
- `test_format_resultados_caracteres_especiales`: texto con emojis, acentos, ñ → no explota
- `test_score_color`: score ≥0.8 → verde, 0.6-0.8 → amarillo, <0.6 → rojo
