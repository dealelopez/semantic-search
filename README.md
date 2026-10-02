# Buscador Semántico Local

Un buscador semántico local para notas en Markdown. En lugar de buscar por palabras exactas, busca por **significado**: si preguntas "cómo funcionan las redes neuronales", encuentra documentos sobre machine learning aunque no contengan exactamente esas palabras.

Todo corre en local: embeddings con Ollama (nomic-embed-text), almacenamiento vectorial con ChromaDB, y una CLI en Python. Sin APIs externas, sin enviar datos a la nube.

## Conceptos clave

### Embedding — texto convertido en números

Un embedding es un vector de 768 números que representa el significado de un texto:

```
"El gato duerme"     → [0.23, -0.41, 0.87, ..., 0.12]  (768 números)
"El felino descansa" → [0.21, -0.39, 0.85, ..., 0.14]  (vectores cercanos!)
"Python es genial"   → [-0.55, 0.73, -0.12, ..., 0.66] (vector lejano)
```

Textos con significado similar producen vectores cercanos en el espacio matemático. Esto permite buscar por significado, no por coincidencia de palabras.

### Distancia coseno — medir cercanía

La distancia coseno mide el ángulo entre dos vectores. En nuestro sistema la convertimos a **score de similitud** (1.0 = idéntico, 0.0 = sin relación). Un resultado con score 0.92 es muy relevante; uno con 0.45 probablemente no.

### Chunking — dividir para conquistar

Un documento largo produce un embedding demasiado difuso (intenta representar demasiados conceptos). Por eso dividimos cada nota en fragmentos de 100-300 tokens antes de generar embeddings. Cada fragmento captura una idea concreta y produce un vector más preciso.

### Modelo de embeddings ≠ LLM generativo

- **nomic-embed-text** (lo que usamos): texto → vector de números. No genera texto.
- **GPT/Llama/Claude**: texto → más texto. Son modelos diferentes con propósitos diferentes.

## Arquitectura

```
┌──────────────────────────────────────────────┐
│            Docker Compose                     │
│                                               │
│  ┌────────────┐     ┌─────────────────────┐  │
│  │  ChromaDB   │◄────│  App Python (CLI)   │  │
│  │  :8000      │     │                     │  │
│  │  vectores   │     │  parsea → trocea    │  │
│  └────────────┘     │  → embede → busca   │  │
│                      └──────────┬──────────┘  │
│                                 │              │
└─────────────────────────────────┼──────────────┘
                                  │ HTTP
                                  ▼
                         ┌──────────────────┐
                         │  Ollama (remoto)  │
                         │  nomic-embed-text │
                         │  :11434           │
                         └──────────────────┘
```

**Componentes:**

| Fichero | Responsabilidad |
|---------|----------------|
| `config.py` | Configuración centralizada (variables de entorno) |
| `models.py` | Dataclasses compartidas (NoteMetadata, Chunk, SearchResult...) |
| `frontmatter_parser.py` | Lee ficheros .md, extrae YAML y body limpio |
| `chunker.py` | Divide texto en fragmentos (headings → párrafos → tamaño fijo) |
| `embeddings.py` | Cliente HTTP para Ollama (genera vectores) |
| `store.py` | Operaciones CRUD con ChromaDB (almacena y busca vectores) |
| `indexer.py` | Orquesta el pipeline (parsear → trocear → embeder → almacenar) |
| `search.py` | Motor de búsqueda y formateo de resultados |
| `cli.py` | Interfaz de línea de comandos |

## Requisitos previos

- **Docker** y **Docker Compose** (para ChromaDB y la app)
- **Ollama** corriendo en un servidor accesible con el modelo instalado:
  ```bash
  # En el servidor con Ollama:
  ollama pull nomic-embed-text
  ```

## Instalación rápida

```bash
# 1. Clonar / copiar el proyecto
cd semantic-search

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env: poner la IP de tu servidor Ollama en OLLAMA_BASE_URL
# Editar .env: poner la ruta a tu vault de notas en NOTES_HOST_DIR
# (ruta absoluta o relativa al proyecto, ej: /Users/usuario/Obsidian/vault).
# El repo no incluye notas de ejemplo: /notas/ está ignorado en git y
# se monta en el contenedor como volumen de solo lectura (:ro).

# 3. Arrancar ChromaDB
docker compose up -d chromadb

# 4. Comprobar conexiones
docker compose run --rm app health

# 5. Indexar todas las notas
docker compose run --rm app index --mode full

# 6. Buscar
docker compose run --rm app search "tu consulta aquí"
```

## Uso

### Comprobar conexiones

```bash
docker compose run --rm app health
```

```
Comprobando conexiones...
  [OK] ChromaDB (chromadb:8000) — conectado (0 chunks indexados)
  [OK] Ollama (http://192.0.2.1:11434) — conectado, modelo nomic-embed-text

  Todo OK. Listo para indexar y buscar.
```

### Indexar notas

```bash
# Primera vez o tras cambiar modelo/chunking:
docker compose run --rm app index --mode full

# Día a día (solo cambios):
docker compose run --rm app index --mode incremental
```

### Búsqueda one-shot

```bash
# Búsqueda simple
docker compose run --rm app search "embeddings y vectores"

# Con filtros
docker compose run --rm app search "configuración de red" --type proyecto
docker compose run --rm app search "inteligencia artificial" --tags ia,ml
docker compose run --rm app search "recetas" -n 10

# Un resultado por nota (el de mayor similitud)
docker compose run --rm app search "redes" --unique

# Salida JSON (para scripting/pipes)
docker compose run --rm app search "embeddings" --json

# Modo debug — muestra filtros, tiempos y distancias
docker compose run --rm app search "redes neuronales" --explain

# Logging detallado (depuración)
docker compose run --rm app -v search "redes"
```

### Búsqueda interactiva (REPL)

```bash
docker compose run --rm app search --interactive
```

```
Buscador semántico — modo interactivo
   Escribe tu consulta y pulsa Enter. 'q' o Ctrl+C para salir.

> cómo funcionan los embeddings
  1. mis-apuntes/ia-basics.md > ## La idea clave
     Similitud: 0.94
     Tags: nota · inteligencia-artificial · embeddings
     Textos con significado similar producen vectores cercanos...

> /type proyecto
   Filtro de tipo activado: proyecto

> servidor casero
  1. mis-apuntes/homelab.md > ## Servicios dockerizados
    Similitud: 0.88
    Tags: proyecto · tecnologia · homelab
    Todos los servicios corren en contenedores Docker...

> /help
   Comandos: /tags, /type, /n, /unique, /clear, /status, /help, q

> q
   ¡Hasta luego!
```

### Ver estado de la colección

```bash
docker compose run --rm app status
```

```
Estado de la colección "notas"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Chunks indexados:  <N> (depende de tu vault)
   Notas indexadas:   <M> (depende de tu vault)
   Colección:         notas
   ChromaDB:          chromadb:8000
   Ollama:            http://192.0.2.1:11434
   Modelo:            nomic-embed-text
```

## Cómo funciona por dentro

El pipeline completo cuando indexas una nota:

1. **Lectura** — escanea `$NOTES_DIR` (tu vault montado en `/notas` como solo lectura) buscando ficheros `.md`
2. **Parsing** — extrae frontmatter YAML (título, tags, fecha, tipo) y body limpio (sin wikilinks de Obsidian)
3. **Chunking** — divide el body en fragmentos de 100-300 tokens usando estrategia híbrida: primero por headings, luego por párrafos, finalmente por tamaño fijo con overlap
4. **Embeddings** — envía cada chunk a Ollama, que devuelve un vector de 768 dimensiones
5. **Almacenamiento** — guarda vector + metadata + texto en ChromaDB con IDs deterministas

Cuando buscas:

1. Tu query se convierte en un vector (con prefijo `search_query:`)
2. ChromaDB busca los vectores más cercanos (algoritmo HNSW, O(log n))
3. Los resultados se ordenan por similitud y se muestran con scores coloreados

## Configuración avanzada

Todas las variables se configuran en `.env` (ver `.env.example` para referencia):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modelo de embeddings |
| `CHROMA_HOST` | `chromadb` | Hostname de ChromaDB |
| `CHROMA_PORT` | `8000` | Puerto de ChromaDB |
| `COLLECTION_NAME` | `notas` | Nombre de la colección vectorial |
| `NOTES_HOST_DIR` | `./notas` | Ruta en el host a tu vault (no hay ejemplos en el repo; apunta a tu carpeta real) |
| `NOTES_DIR` | `/notas` | Ruta dentro del contenedor (no cambiar) |
| `NOTES_IGNORE_DIRS` | `.obsidian,.trash,.git,_templates` | Directorios a ignorar (separados por coma) |
| `MAX_CHUNK_TOKENS` | `300` | Tamaño máximo de un chunk |
| `MIN_CHUNK_TOKENS` | `100` | Tamaño mínimo (se agrupan los pequeños) |
| `OVERLAP_TOKENS` | `50` | Solapamiento entre chunks |
| `EMBED_BATCH_SIZE` | `50` | Textos por petición a Ollama |
| `EMBED_TIMEOUT` | `120` | Timeout en segundos por petición |
| `DEFAULT_N_RESULTS` | `5` | Resultados por defecto en búsquedas |

**Tips:**
- Si cambias el modelo de embeddings, haz `index --mode full` (los vectores de modelos diferentes son incompatibles)
- Si cambias los parámetros de chunking, también necesitas `full` (los chunks son diferentes)
- Si Ollama va lento, baja `EMBED_BATCH_SIZE` a 20-30

## Desarrollo local

```bash
# Crear entorno virtual
uv venv --python 3.12 .venv
source .venv/bin/activate

# Instalar dependencias de desarrollo
uv pip install -r requirements-dev.txt

# Ejecutar tests (no necesitan Docker ni Ollama)
python -m pytest -v -m "not integration"

# Ejecutar la CLI localmente (necesita ChromaDB y Ollama)
export CHROMA_HOST=localhost
export NOTES_DIR=/ruta/a/tu/vault
python -m src.cli health
```

### Hook pre-commit anti-PII

Un hook de git (Python, sin dependencias) que bloquea el commit si detecta
claves privadas, tokens, emails, rutas locales con tu usuario o literales de
`scripts/pii_denylist.txt`:

```bash
scripts/install_hooks.sh   # una vez por clon
```

Revisión manual: `python3 scripts/check_no_pii.py` (staged) o
`python3 scripts/check_no_pii.py --all` (todo el repo). Tus valores reales van
en `scripts/pii_denylist.local.txt` (ignorado por git, nunca se commitea).

## Troubleshooting

**"No se puede conectar a Ollama"**
- Verifica que Ollama está arrancado: `curl http://<IP>:11434/`
- Comprueba que acepta conexiones remotas: `OLLAMA_HOST=0.0.0.0 ollama serve`
- Revisa el firewall

**"No se puede conectar a ChromaDB"**
- `docker compose up -d chromadb`
- Espera al healthcheck: `docker compose ps` (debe mostrar "healthy")

**"Modelo no disponible"**
- `ollama pull nomic-embed-text` en el servidor con Ollama

**"Resultados de búsqueda malos"**
- ¿Has indexado? `docker compose run --rm app status`
- Prueba con otros términos o sinónimos
- Revisa que las notas tienen suficiente contenido (párrafos cortos generan embeddings pobres)

**"Indexación muy lenta"**
- Reduce `EMBED_BATCH_SIZE` si Ollama tiene poca RAM
- nomic-embed-text en CPU tarda ~1-2 segundos por batch de 50 textos
- Para 400 notas: ~3-8 minutos en full, segundos en incremental

## Licencia

Copyright (C) 2026 Ale López. Publicado bajo **GPL-3.0-or-later** (ver `LICENSE`).
