# Task 1: Scaffolding del proyecto y Docker Compose

## Estado: [OK] COMPLETADO

## Objetivo

Crear la estructura de directorios, `docker-compose.yml`, `Dockerfile`, `example.env`, `requirements.txt`, `.gitignore` y `src/config.py`. Verificar que ChromaDB arranca y responde al healthcheck.

## Ficheros a crear

| Fichero | Descripción |
|---------|-------------|
| `docker-compose.yml` | Dos servicios: `chromadb` + `app` |
| `Dockerfile` | Imagen Python 3.12-slim para la app |
| `example.env` | Plantilla de variables de entorno |
| `.gitignore` | Excluir `.env`, `chroma-data/`, `__pycache__/` |
| `requirements.txt` | Dependencias pinneadas |
| `src/__init__.py` | Paquete principal con descripción del proyecto |
| `src/config.py` | Lee variables de entorno, define constantes de chunking/embeddings |
| `tests/__init__.py` | Paquete de tests |
| `notas/.gitkeep` | Punto de montaje para las notas MD |

## Detalle de implementación

### docker-compose.yml

Dos servicios:

- **chromadb**: imagen `chromadb/chroma:1.5.3`, puerto 8000, volumen `./chroma-data:/data`, healthcheck en `/api/v2/heartbeat`, límite de memoria 512M.
- **app**: build desde `Dockerfile`, env_file `.env`, depends_on `chromadb` (healthy), volumen `./notas:/notas:ro` (solo lectura), `extra_hosts` para `host.docker.internal`.

### Dockerfile

- Base: `python:3.12-slim`
- `PYTHONDONTWRITEBYTECODE=1` y `PYTHONUNBUFFERED=1`
- Copia `requirements.txt` primero (aprovecha caché de capas Docker)
- `pip install --no-cache-dir -r requirements.txt`
- Copia `src/` al contenedor
- Entrypoint: `python -m src.cli`
- CMD por defecto: `--help`

### requirements.txt

```
httpx==0.28.1              # Cliente HTTP para Ollama
chromadb-client==1.5.3     # Cliente Python para ChromaDB en modo servidor
python-frontmatter==1.1.0  # Parser de frontmatter YAML
PyYAML==6.0.2              # Parser YAML
rich==14.0.0               # Output bonito en terminal
pytest==8.3.5              # Testing
```

### src/config.py

Variables de entorno con valores por defecto:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modelo de embeddings |
| `CHROMA_HOST` | `chromadb` | Hostname de ChromaDB (DNS Docker) |
| `CHROMA_PORT` | `8000` | Puerto de ChromaDB |
| `COLLECTION_NAME` | `notas` | Nombre de la colección vectorial |
| `NOTES_DIR` | `/notas` | Directorio de notas dentro del contenedor |
| `MAX_CHUNK_TOKENS` | `300` | Tamaño máximo de chunk |
| `MIN_CHUNK_TOKENS` | `100` | Tamaño mínimo de chunk |
| `OVERLAP_TOKENS` | `50` | Solapamiento entre chunks |
| `EMBED_BATCH_SIZE` | `50` | Textos por petición a Ollama |
| `EMBED_TIMEOUT` | `120` | Timeout en segundos por petición |
| `EMBED_RETRIES` | `3` | Reintentos con backoff exponencial |
| `DEFAULT_N_RESULTS` | `5` | Resultados por búsqueda |
| `PREVIEW_MAX_CHARS` | `200` | Longitud del preview en resultados |

Cada variable lleva un docstring extenso explicando qué hace y por qué tiene ese valor por defecto.

## Verificación

```bash
docker compose up -d chromadb
curl http://localhost:8000/api/v2/heartbeat  # → 200 OK
docker compose down
```
