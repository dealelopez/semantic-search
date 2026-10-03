# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# config.py — Configuración centralizada del buscador semántico
# =============================================================================
#
# Todas las configuraciones se leen de variables de entorno (definidas en .env).
# Esto permite cambiar el comportamiento sin tocar código, y hace que el
# proyecto sea portable entre máquinas (cada una con su propio .env).
#
# Las constantes de chunking y embeddings se definen aquí como valores por
# defecto que se pueden ajustar vía entorno si se necesita experimentar.
# =============================================================================

import logging
import os
from pathlib import Path


# =============================================================================
# HELPERS — Lectura y validación de variables de entorno
# =============================================================================
# Funciones internas que convierten variables de entorno a su tipo correcto
# con mensajes de error claros si el valor es inválido. Evitan que un typo
# en .env provoque un traceback críptico de Python.
# =============================================================================

def _env_int(name: str, default: int) -> int:
    """Lee una variable de entorno como entero, con mensaje claro si falla."""
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(
            f"Error de configuración: {name}={raw!r} no es un entero válido. "
            f"Revisa tu fichero .env."
        )


def _env_log_level(name: str, default: str) -> int:
    """Lee una variable de entorno como nivel de logging, con validación."""
    raw = os.getenv(name, default).upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        valid = "DEBUG, INFO, WARNING, ERROR, CRITICAL"
        raise SystemExit(
            f"Error de configuración: {name}={raw!r} no es un nivel de log válido. "
            f"Valores aceptados: {valid}. Revisa tu fichero .env."
        )
    return level


# =============================================================================
# LOGGING — Nivel de detalle de los mensajes de log
# =============================================================================
# Controla cuánta información se muestra en la terminal durante la ejecución.
# Útil para depurar problemas sin modificar código.
#
# Valores válidos (de más a menos verboso):
#   DEBUG    → Todo: peticiones HTTP, payloads, tiempos
#   INFO     → Flujo normal: "indexando nota X", "búsqueda completada"
#   WARNING  → Solo problemas potenciales (default)
#   ERROR    → Solo errores que impiden continuar
# =============================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING").upper()
"""
Nivel de logging. Cambia a DEBUG para depurar problemas con Ollama o ChromaDB.
Ejemplo: LOG_LEVEL=DEBUG docker compose run app index --mode full
"""

def setup_logging(verbose: bool = False) -> None:
    """
    Configura logging global. Llamar desde main() de la CLI, no al importar.

    Args:
        verbose: Si True, fuerza nivel DEBUG (flag -v de la CLI).
    """
    level = logging.DEBUG if verbose else _env_log_level("LOG_LEVEL", "WARNING")
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# =============================================================================
# OLLAMA — Servidor de modelos de IA
# =============================================================================
# Ollama es un servidor local que ejecuta modelos de IA. En nuestro caso,
# lo usamos exclusivamente para generar embeddings (convertir texto a vectores).
#
# El servidor Ollama corre en otro equipo de la red local y se comunica
# por HTTP. La URL incluye protocolo, IP y puerto.
# =============================================================================

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
"""URL base del servidor Ollama. Ejemplo: http://192.0.2.1:11434 (IP de documentación)"""

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
"""
Modelo de embeddings a usar. Opciones recomendadas:

  - nomic-embed-text (~274MB, 768 dims)
    Mejor equilibrio calidad/velocidad para CPU. Buen soporte multilingüe.
    IMPORTANTE: requiere prefijos "search_document: " y "search_query: "
    para distinguir si el texto es un documento a indexar o una consulta.

  - all-minilm (~45MB, 384 dims)
    El más rápido y ligero. Pero peor rendimiento en español.

  - mxbai-embed-large (~670MB, 1024 dims)
    Mejor calidad, pero significativamente más lento en CPU.
"""


# =============================================================================
# CHROMADB — Base de datos vectorial
# =============================================================================
# ChromaDB almacena los embeddings y permite buscar por similitud.
# Corre en un contenedor Docker separado y se comunica por HTTP.
#
# Dentro de Docker Compose, el nombre del servicio ("chromadb") funciona
# como hostname gracias al DNS interno de Docker.
# =============================================================================

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "chromadb")
"""Hostname del servidor ChromaDB. 'chromadb' dentro de Docker Compose, 'localhost' fuera."""

CHROMA_PORT: int = _env_int("CHROMA_PORT", 8000)
"""Puerto del servidor ChromaDB."""

COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "notas")
"""
Nombre de la colección en ChromaDB. Una colección es el equivalente a una
"tabla" en bases de datos tradicionales: agrupa todos los vectores (embeddings)
junto con sus metadatos (tags, título, fecha, etc.) y el texto original.

Puedes tener múltiples colecciones si quieres indexar diferentes corpus
por separado (ej: "notas", "documentacion", "emails").
"""


# =============================================================================
# NOTAS — Directorio de ficheros Markdown
# =============================================================================

NOTES_DIR: str = os.getenv("NOTES_DIR", "./notas")
"""
Ruta al directorio que contiene las notas Markdown.

Dentro de Docker es /notas (montado como volumen read-only, definido en .env).
Fuera de Docker (desarrollo local), el default es ./notas (relativo al CWD).

Para desarrollo local, puedes hacer:
  export NOTES_DIR=./notas    (o la ruta a tu vault de Obsidian)
  python -m src.cli search "mi consulta"
"""

# Directorios a ignorar al escanear notas.
# Configurable vía NOTES_IGNORE_DIRS en .env (separados por coma).
# Acepta nombres exactos ("4-meta") y globs fnmatch (".*" = todas las ocultas).
_DEFAULT_IGNORE_DIRS = ".obsidian,.trash,.git,_templates,.stversions,.stfolder,4-meta,.*"
NOTES_IGNORE_PATTERNS: list[str] = [
    d.strip()
    for d in os.getenv("NOTES_IGNORE_DIRS", _DEFAULT_IGNORE_DIRS).split(",")
    if d.strip()
]
"""
Nombres de directorios a ignorar durante el escaneo recursivo de notas.
Cualquier directorio cuyo nombre coincida con un patrón se salta entero.
Acepta globs estilo fnmatch: ".*" ignora todas las carpetas ocultas
(.obsidian, .app, .vscode, .stversions, .varios, futuras...).
Configurable vía variable de entorno NOTES_IGNORE_DIRS (separados por coma).
Default: .obsidian, .trash, .git, _templates, .stversions, .stfolder, 4-meta, .*
"""


# =============================================================================
# CHUNKING — Parámetros de troceado del texto
# =============================================================================
# El "chunking" es el proceso de dividir un documento largo en fragmentos
# más pequeños antes de generar embeddings.
#
# ¿Por qué trocear?
# Un modelo de embeddings convierte un texto en un vector de N dimensiones
# (por ejemplo, 768 números). Ese vector debe capturar el "significado"
# del texto. Si el texto es muy largo (un documento entero), el vector
# resultante es demasiado difuso — intenta representar demasiados conceptos
# a la vez y no captura bien ninguno. Un chunk de 100-300 palabras sobre
# un tema concreto produce un vector mucho más preciso.
#
# ¿Qué pasa si el chunk es muy pequeño?
# Pierde contexto. Una frase suelta como "Esto funciona bien" no tiene
# suficiente información para generar un embedding útil. Necesitamos
# al menos un párrafo (~150 tokens) para que el vector sea significativo.
#
# Los valores por defecto son un buen punto de partida. Se pueden ajustar
# experimentando con los resultados de búsqueda.
# =============================================================================

MAX_CHUNK_TOKENS: int = _env_int("MAX_CHUNK_TOKENS", 300)
"""
Tamaño máximo de un chunk en tokens (aproximados).
Si una sección del documento supera este límite, se subdivide.

300 tokens ≈ 200-250 palabras en español. Es suficiente para capturar
una idea completa sin ser tan largo que el embedding se diluya.

El modelo nomic-embed-text acepta hasta 8192 tokens, pero embeddings
de textos más cortos y enfocados producen mejores resultados de búsqueda.
"""

MIN_CHUNK_TOKENS: int = _env_int("MIN_CHUNK_TOKENS", 100)
"""
Tamaño mínimo de un chunk en tokens (aproximados).
Si un fragmento es más pequeño, se agrupa con el siguiente hasta alcanzar
este mínimo. Esto evita chunks tan cortos que no tienen suficiente
información semántica para generar un embedding útil.
"""

OVERLAP_TOKENS: int = _env_int("OVERLAP_TOKENS", 50)
"""
Tokens de solapamiento entre chunks consecutivos cuando se subdivide
por tamaño fijo.

¿Para qué sirve el overlap?
Imagina que una idea clave cae justo en el borde entre dos chunks.
Sin overlap, esa idea se parte en dos y ninguno de los chunks la captura
completa. Con overlap, los últimos ~50 tokens del chunk N se repiten al
inicio del chunk N+1, asegurando que la idea aparece completa en al menos
uno de los dos.

El trade-off: más overlap = mejor cobertura pero más chunks (y más
embeddings que generar y almacenar).
"""


# =============================================================================
# EMBEDDINGS — Parámetros de generación de vectores
# =============================================================================
# Estos parámetros controlan cómo se comunica la app con Ollama para
# generar embeddings. Son importantes para el rendimiento en hardware
# limitado (CPU-only).
# =============================================================================

EMBED_BATCH_SIZE: int = _env_int("EMBED_BATCH_SIZE", 50)
"""
Número de textos a enviar a Ollama en cada petición HTTP.

Ollama acepta una lista de textos y devuelve una lista de vectores.
Enviar de golpe 2000 chunks saturaría la CPU del servidor (hardware modesto).
Procesamos en lotes de 50 para que Ollama pueda ir generando vectores
sin quedarse sin RAM ni provocar timeouts.

Ajustar según la RAM disponible: con 16GB para Ollama, 50 es conservador.
Se podría subir a 100 si el rendimiento es estable.
"""

EMBED_TIMEOUT: int = _env_int("EMBED_TIMEOUT", 120)
"""
Timeout en segundos para cada petición HTTP a Ollama.

En CPU (sin GPU), generar embeddings de un lote de 50 textos puede
tardar 30-60 segundos. Ponemos 120s para dar margen. Si ves timeouts
frecuentes, sube este valor o baja EMBED_BATCH_SIZE.
"""

EMBED_RETRIES: int = _env_int("EMBED_RETRIES", 3)
"""
Número de reintentos si una petición a Ollama falla.

Usa backoff exponencial: tras fallo 1 → espera 1s, fallo 2 → espera 2s,
fallo 3 → espera 4s. Fórmula: wait = 2^attempt (con attempt empezando en 0).

En CPU, a veces la primera petición tarda más porque Ollama necesita
cargar el modelo en RAM. Los reintentos cubren este caso.
"""


# =============================================================================
# BÚSQUEDA — Parámetros por defecto
# =============================================================================

DEFAULT_N_RESULTS: int = _env_int("DEFAULT_N_RESULTS", 5)
"""Número de resultados a devolver por defecto en cada búsqueda."""

PREVIEW_MAX_CHARS: int = _env_int("PREVIEW_MAX_CHARS", 200)
"""Longitud máxima del preview de texto en los resultados de búsqueda."""

MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.0"))
"""
Umbral mínimo de similitud (0.0 a 1.0). Resultados por debajo se descartan.
0.0 = sin filtro (default). 0.5 = solo resultados medianamente relevantes.
Ajustar según la calidad deseada. Configurable también con --min-score en la CLI.
"""


# =============================================================================
# VALIDACIÓN — Comprobación de la configuración
# =============================================================================

def validate_config() -> list[str]:
    """
    Valida la configuración y devuelve una lista de problemas encontrados.

    Returns:
        Lista de strings con problemas. Vacía si todo está correcto.
    """
    issues: list[str] = []

    if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
        issues.append(
            f"OLLAMA_BASE_URL debe empezar con http:// o https:// "
            f"(actual: {OLLAMA_BASE_URL!r})"
        )

    if not EMBEDDING_MODEL:
        issues.append("EMBEDDING_MODEL no puede estar vacío")

    if MAX_CHUNK_TOKENS <= MIN_CHUNK_TOKENS:
        issues.append(
            f"MAX_CHUNK_TOKENS ({MAX_CHUNK_TOKENS}) debe ser mayor que "
            f"MIN_CHUNK_TOKENS ({MIN_CHUNK_TOKENS})"
        )

    if OVERLAP_TOKENS >= MAX_CHUNK_TOKENS:
        issues.append(
            f"OVERLAP_TOKENS ({OVERLAP_TOKENS}) debe ser menor que "
            f"MAX_CHUNK_TOKENS ({MAX_CHUNK_TOKENS})"
        )

    if not (0.0 <= MIN_SCORE <= 1.0):
        issues.append(
            f"MIN_SCORE debe estar entre 0.0 y 1.0 (actual: {MIN_SCORE})"
        )

    return issues
