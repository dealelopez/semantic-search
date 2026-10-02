# Task 8: CLI — One-shot y modo REPL interactivo

## Estado: ✅ HECHO

## Objetivo

Implementar `cli.py` como entrypoint principal con subcomandos: `index` (full/incremental), `search` (one-shot y REPL), `status` (info de la colección), `health` (comprobación de conexiones).

## Ficheros a crear/modificar

| Fichero | Acción | Descripción |
|---------|--------|-------------|
| `src/cli.py` | CREAR | Entrypoint CLI con argparse |
| `docker-compose.yml` | MODIFICAR | Ajustar entrypoint para uso cómodo |

## Detalle de implementación

### src/cli.py — Estructura general

```python
"""
Punto de entrada del buscador semántico.

Uso desde Docker:
  docker compose run app health                      # Comprueba conexiones
  docker compose run app index --mode full            # Indexa todas las notas
  docker compose run app index --mode incremental     # Solo nuevos/modificados
  docker compose run app search "mi consulta"         # Búsqueda one-shot
  docker compose run app search --interactive         # Modo REPL
  docker compose run app search "ia" --type nota      # Filtrar por tipo
  docker compose run app search "ia" --tags ml,deep   # Filtrar por tags
  docker compose run app status                       # Info de la colección
"""
```

### Subcomandos

**health** — Comprueba conexiones

```
$ docker compose run app health

Comprobando conexiones...
  [OK] ChromaDB (chromadb:8000) — conectado
  [OK] Ollama (http://192.0.2.1:11434) — conectado, modelo nomic-embed-text disponible
  
  Todo OK. Listo para indexar y buscar.
```

```
$ docker compose run app health

Comprobando conexiones...
  [OK] ChromaDB (chromadb:8000) — conectado
  [ERROR] Ollama (http://192.0.2.1:11434) — no se puede conectar
  
  [WARN] Revisa que Ollama está arrancado y accesible desde Docker.
```

Implementación:
1. `store.collection_stats()` → si responde, ChromaDB OK
2. `embedder.health_check()` → si responde, Ollama OK
3. Opcionalmente: verificar que el modelo está disponible haciendo un embed de prueba

**index** — Indexación

```
$ docker compose run app index --mode full

Indexación COMPLETA — se reindexará todo desde cero
   Directorio: /notas (312 ficheros .md encontrados)
   
   Indexando notas... ━━━━━━━━━━━━━━━━━━━━━━ 312/312 100%
   
Reporte:
   Notas procesadas: 312
   Chunks creados:   1,847
   Errores:          2
   Duración:         4m 23s
   
[WARN] Errores (2):
   - nota-corrupta.md: frontmatter YAML malformado
   - nota-vacia.md: sin contenido para indexar
```

```
$ docker compose run app index --mode incremental

Indexación INCREMENTAL — solo cambios
   Directorio: /notas
   
   Analizando cambios...
   → 3 nuevas, 1 modificada, 0 eliminadas, 308 sin cambios
   
   Procesando cambios... ━━━━━━━━━━━━━━━ 4/4 100%
   
Reporte:
   Notas procesadas: 4 (3 nuevas, 1 modificada)
   Notas sin cambios: 308
   Notas eliminadas:  0
   Chunks creados:    22
   Duración:          12s
```

Argumentos:
- `--mode full|incremental` (requerido)

**search** — Búsqueda

```
$ docker compose run app search "cómo funcionan los embeddings"

Resultados para: "cómo funcionan los embeddings"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. embeddings-intro.md > ## Qué es un embedding
    Similitud: 0.94
    Tags: nota · ia · embeddings
    Un embedding es una forma de convertir un texto en una
       lista de números (un vector) que representa su significado...

 2. ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5 resultados en 0.34s
```

Argumentos:
- `query` (posicional, requerido en modo one-shot)
- `--interactive` / `-i` — modo REPL
- `--tags tag1,tag2` — filtrar por tags (separados por coma en la CLI, se convierten a lista internamente)
- `--type tipo` — filtrar por tipo de nota (nota, proyecto, receta, etc.)
- `-n N` — número de resultados (default 5)

**search --interactive** — Modo REPL

```
$ docker compose run app search --interactive

Buscador semántico — modo interactivo
   Escribe tu consulta y pulsa Enter. 'q' o Ctrl+C para salir.
   Opciones: /tags tag1,tag2  /type nota  /n 10  /help
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> cómo funcionan los embeddings

 1. embeddings-intro.md > ## Qué es un embedding
    Similitud: 0.94
    ...

> /type proyecto
   Filtro de tipo activado: proyecto

> estado del pipeline de datos
 1. pipeline-datos.md > ## Estado actual
    Similitud: 0.88
    ...

> /tags ia,ml
   Filtro de tags activado: ia, ml

> /help
   Comandos disponibles:
     /tags tag1,tag2  — Filtrar por tags (vacío para quitar filtro)
     /type tipo       — Filtrar por tipo de nota (vacío para quitar)
     /n N             — Cambiar número de resultados
     /clear           — Limpiar filtros
     /status          — Mostrar estado de la colección
     /help            — Mostrar esta ayuda
     q                — Salir

> q
   ¡Hasta luego!
```

Implementación del REPL:

```python
def _interactive_mode(search_engine, n_results, filter_tags, filter_type):
    console = Console()
    console.print("Buscador semántico — modo interactivo")
    console.print("   Escribe tu consulta. 'q' o Ctrl+C para salir. /help para comandos.")
    console.print("━" * 60)

    while True:
        try:
            query = console.input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n   ¡Hasta luego!")
            break

        if not query:
            continue
        if query.lower() == "q":
            console.print("   ¡Hasta luego!")
            break
        if query.startswith("/"):
            # Procesar comando REPL (/tags, /type, /n, /clear, /status, /help)
            _handle_repl_command(query, ...)
            continue

        # Ejecutar búsqueda
        results = search_engine.search(query, n_results, filter_tags, filter_type)
        search_engine.format_results(query, results, elapsed)
```

**status** — Estado de la colección

```
$ docker compose run app status

Estado de la colección "notas"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Chunks indexados:  1,847
   Notas indexadas:   312
   Colección:         notas
   ChromaDB:          chromadb:8000
   Ollama:            http://192.0.2.1:11434
   Modelo:            nomic-embed-text
```

### Factory function — Creación de dependencias

```python
def _create_dependencies():
    """
    Crea todas las dependencias del sistema a partir de la configuración.
    Esta función es el punto central donde se "ensambla" la aplicación.
    """
    embedder = OllamaEmbedder(
        base_url=config.OLLAMA_BASE_URL,
        model=config.EMBEDDING_MODEL,
    )
    store = VectorStore(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        collection_name=config.COLLECTION_NAME,
    )
    chunker = HybridChunker()
    indexer = Indexer(embedder=embedder, store=store, chunker=chunker)
    search_engine = SearchEngine(embedder=embedder, store=store)
    return embedder, store, chunker, indexer, search_engine
```

### Manejo de errores amigable

- Si ChromaDB no responde: `"[ERROR] No se puede conectar a ChromaDB en {host}:{port}. ¿Está arrancado? Ejecuta: docker compose up -d chromadb"`
- Si Ollama no responde: `"[ERROR] No se puede conectar a Ollama en {url}. ¿Está arrancado el servidor?"`
- Si el directorio de notas no existe o está vacío: `"[WARN] No se encontraron ficheros .md en {dir}. ¿Has montado las notas?"`

### Comentarios didácticos a incluir

- Cómo argparse maneja subcomandos y por qué es mejor que parsear sys.argv manualmente
- Por qué la factory function centraliza la creación de dependencias (patrón de inyección de dependencias)
- Por qué el REPL mantiene estado (filtros) entre consultas — mejora la experiencia de exploración

### Tests

- `test_argparse_search_oneshot`: parsear `["search", "mi query"]` → args correctos
- `test_argparse_search_interactive`: parsear `["search", "--interactive"]` → args.interactive == True
- `test_argparse_index_full`: parsear `["index", "--mode", "full"]` → args.mode == "full"
- `test_argparse_search_con_filtros`: parsear `["search", "query", "--tags", "ia,ml", "--type", "nota", "-n", "10"]` → filtros correctos
- `test_create_dependencies`: verificar que la factory crea los objetos correctos (sin conectar)
