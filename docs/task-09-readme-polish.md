# Task 9: README didáctico y pulido final

## Estado: ⬜ PENDIENTE

## Objetivo

Crear un `README.md` completo que sirva como guía de uso y como documento didáctico sobre embeddings y búsqueda semántica. Crear notas de ejemplo en `notas/` para testing. Verificar el flujo end-to-end completo.

## Ficheros a crear

| Fichero | Descripción |
|---------|-------------|
| `README.md` | Guía de uso + documento didáctico |
| `notas/ejemplo-embeddings.md` | Nota de ejemplo sobre embeddings |
| `notas/ejemplo-proyecto-homelab.md` | Nota tipo proyecto |
| `notas/ejemplo-receta-tortilla.md` | Nota tipo receta |
| `notas/ejemplo-reunion-equipo.md` | Nota tipo reunión |
| `notas/ejemplo-nota-sin-frontmatter.md` | Nota sin frontmatter |

## Detalle de implementación

### README.md — Estructura

#### 1. Qué es esto (1 párrafo)

Un buscador semántico local para notas en Markdown. En lugar de buscar por palabras exactas, busca por significado: si preguntas "cómo funcionan las redes neuronales", encuentra documentos sobre machine learning aunque no contengan exactamente esas palabras.

#### 2. Conceptos clave (sección didáctica)

Breve explicación con diagramas ASCII:

- **Embedding**: vector numérico que representa el significado de un texto
  ```
  "El gato duerme" → [0.23, -0.41, 0.87, ..., 0.12]  (768 números)
  "El felino descansa" → [0.21, -0.39, 0.85, ..., 0.14]  (vectores cercanos!)
  "Python es un lenguaje" → [-0.55, 0.73, -0.12, ..., 0.66]  (vector lejano)
  ```
- **Distancia coseno**: cómo se mide la cercanía entre vectores
- **Chunking**: por qué un documento largo se divide en fragmentos antes de generar embeddings
- **Modelo de embeddings vs LLM generativo**: por qué son cosas diferentes

#### 3. Arquitectura

Diagrama de componentes:
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

#### 4. Requisitos previos

- Docker y Docker Compose
- Ollama corriendo en un servidor accesible con `nomic-embed-text` instalado:
  ```bash
  # En el servidor con Ollama:
  ollama pull nomic-embed-text
  ```

#### 5. Instalación rápida

```bash
# 1. Clonar / copiar el proyecto
cd semantic-search

# 2. Configurar variables de entorno
cp example.env .env
# Editar .env: poner la IP de tu servidor Ollama

# 3. Poner tus notas en ./notas/
cp -r /ruta/a/tus/notas/*.md ./notas/

# 4. Arrancar
docker compose up -d

# 5. Comprobar conexiones
docker compose run app health

# 6. Indexar
docker compose run app index --mode full

# 7. Buscar
docker compose run app search "tu consulta aquí"
```

#### 6. Uso — Ejemplos de cada comando

Ejemplos completos de:
- `health` — comprobar que todo funciona
- `index --mode full` — primera indexación
- `index --mode incremental` — actualizar tras añadir notas
- `search "query"` — búsqueda one-shot
- `search --interactive` — modo REPL
- `search "query" --type proyecto` — filtrar por tipo
- `search "query" --tags ia,ml` — filtrar por tags
- `search "query" -n 10` — más resultados
- `status` — ver info de la colección

#### 7. Cómo funciona por dentro

Walkthrough del pipeline:
1. **Lectura**: escanea `notas/` buscando ficheros `.md`
2. **Parsing**: extrae frontmatter YAML (título, tags, fecha, tipo) y body limpio
3. **Chunking**: divide el body en fragmentos de 100-300 tokens usando estrategia híbrida
4. **Embeddings**: envía cada chunk a Ollama, que devuelve un vector de 768 dimensiones
5. **Almacenamiento**: guarda vector + metadata + texto en ChromaDB
6. **Búsqueda**: convierte tu query en vector, busca los más cercanos, muestra resultados

#### 8. Configuración avanzada

Tabla con todas las variables de entorno, sus valores por defecto, y qué hacen. Incluye tips como:
- Cambiar modelo de embeddings (y cuándo hacer full reindex)
- Ajustar parámetros de chunking para experimentar
- Subir/bajar batch size según la RAM disponible

#### 9. Troubleshooting

Problemas comunes y soluciones:
- "No se puede conectar a Ollama" → verificar IP, firewall, `OLLAMA_HOST=0.0.0.0`
- "No se puede conectar a ChromaDB" → `docker compose up -d chromadb`, verificar healthcheck
- "Modelo no disponible" → `ollama pull nomic-embed-text`
- "Resultados de búsqueda malos" → revisar chunking, probar otro modelo, verificar que las notas tienen suficiente contenido
- "Indexación muy lenta" → reducir batch size, verificar RAM del servidor, considerar modelo más ligero

### Notas de ejemplo

5 notas que cubren diferentes tipos del sistema frontmatter v2:

**ejemplo-embeddings.md** (tipo: nota)
- Frontmatter con tags: nota, ia, embeddings
- Contenido sobre qué son los embeddings, con secciones h2
- ~200 palabras

**ejemplo-proyecto-homelab.md** (tipo: proyecto)
- Frontmatter con status: activo, tags: proyecto, tecnologia
- Contenido sobre un proyecto de homelab
- ~150 palabras

**ejemplo-receta-tortilla.md** (tipo: receta)
- Frontmatter con tags: receta, url, rating
- Contenido con ingredientes y preparación
- ~100 palabras

**ejemplo-reunion-equipo.md** (tipo: reunion)
- Frontmatter con date, people, org
- Contenido con resumen de reunión
- ~120 palabras

**ejemplo-nota-sin-frontmatter.md** (sin tipo)
- Sin bloque YAML
- Markdown plano con headings
- ~80 palabras

### Verificación end-to-end

Ejecutar el flujo completo desde cero:

```bash
docker compose down -v          # Limpiar todo
docker compose up -d            # Arrancar
docker compose run app health   # Verificar conexiones
docker compose run app index --mode full   # Indexar las 5 notas ejemplo
docker compose run app search "embeddings y vectores"  # Búsqueda
docker compose run app search --interactive  # REPL
docker compose run app status   # Verificar stats (5 notas, N chunks)
```

Verificar que:
- Health muestra todo OK
- Indexación procesa 5 notas sin errores
- Búsqueda devuelve resultados relevantes (la nota de embeddings debería ser el top result)
- REPL funciona y permite búsquedas consecutivas
- Status muestra los conteos correctos
