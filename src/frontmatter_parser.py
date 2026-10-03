# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# frontmatter_parser.py — Extracción de metadatos YAML de ficheros Markdown
# =============================================================================
#
# Qué es frontmatter:
# ---
# Muchos sistemas de notas (Obsidian, Jekyll, Hugo, Astro) usan un bloque
# YAML al inicio del fichero Markdown, delimitado por "---", para almacenar
# metadatos: título, tags, fecha, etc. Este bloque se llama "frontmatter".
#
# Por qué es valioso para búsqueda semántica:
# ---
# Los metadatos del frontmatter permiten FILTRAR resultados además de buscar
# por similitud. Puedes buscar "redes neuronales" pero solo en notas de tipo
# "proyecto", o solo con el tag "ia". Esto combina lo mejor de la búsqueda
# semántica (encontrar por significado) con filtros estructurados (precisión).
#
# Qué hace este módulo:
# ---
# 1. Lee un fichero .md
# 2. Separa el frontmatter YAML del cuerpo markdown
# 3. Construye un NoteMetadata con los campos relevantes
# 4. Limpia el body (elimina wikilinks de Obsidian)
# 5. Devuelve un ParsedNote listo para chunking
# =============================================================================

import fnmatch
import logging
import re
from pathlib import Path

import frontmatter

from src import config
from src.models import NoteMetadata, ParsedNote

logger = logging.getLogger(__name__)


# =============================================================================
# REGEX — Limpieza de wikilinks de Obsidian
# =============================================================================
# Obsidian usa una sintaxis propia para enlaces internos: [[nombre de nota]].
# También soporta alias: [[nota destino|texto visible]].
#
# ¿Por qué limpiarlos?
# Los wikilinks añaden ruido a los embeddings. El texto "[[RAG — Retrieval-
# Augmented Generation]]" no aporta más significado semántico que "RAG —
# Retrieval-Augmented Generation" sin los corchetes. Pero sí confunde al
# modelo de embeddings, que puede tratar los corchetes como parte del significado.
#
# La regex captura dos casos:
#   [[texto]]           → texto
#   [[target|display]]  → display  (solo el texto visible)
# =============================================================================

_WIKILINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")


def _clean_wikilinks(text: str) -> str:
    """
    Reemplaza wikilinks de Obsidian por su texto visible.

    Ejemplos:
        [[Machine Learning]]              → Machine Learning
        [[ML Basics|Machine Learning]]    → Machine Learning
        [[RAG — Retrieval-Augmented Generation]] → RAG — Retrieval-Augmented Generation
    """
    return _WIKILINK_RE.sub(r"\1", text)


# =============================================================================
# PARSING — Lectura y extracción de metadatos
# =============================================================================

def parse_note(filepath: Path) -> ParsedNote:
    """
    Lee un fichero Markdown y extrae su frontmatter YAML y cuerpo limpio.

    Args:
        filepath: Ruta al fichero .md (absoluta o relativa).

    Returns:
        ParsedNote con los metadatos estructurados y el body limpio.

    El parser es tolerante con notas sin frontmatter (~2% del corpus):
    genera metadatos por defecto a partir del nombre del fichero.
    También maneja frontmatter malformado con un fallback seguro.
    """
    logger.debug("Parseando nota: %s", filepath)

    # python-frontmatter separa automáticamente el bloque YAML del body.
    # Si no hay bloque --- al inicio, post.metadata será un dict vacío
    # y post.content será todo el fichero.
    post = frontmatter.load(str(filepath))

    meta = post.metadata  # dict con los campos YAML (puede estar vacío)
    body = post.content    # string con el body markdown (sin el bloque YAML)

    # --- Extraer campos core del frontmatter ---

    # Título: del YAML, o nombre del fichero sin extensión como fallback.
    title = meta.get("title", filepath.stem)
    # Asegurar que title es string (a veces YAML lo parsea como otro tipo).
    title = str(title) if title is not None else filepath.stem

    # Tags: lista de strings. Si el campo no existe o no es lista, vacío.
    # Se sanean: se descartan None/vacíos y se admite string "a, b".
    raw_tags = meta.get("tags", [])
    if isinstance(raw_tags, list):
        tags = [str(t).strip() for t in raw_tags if t is not None and str(t).strip()]
    elif isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    else:
        tags = []

    # Tipo de nota: primer tag (sistema v2 del usuario).
    # "sin-tipo" si no hay tags.
    note_type = tags[0] if tags else "sin-tipo"

    # Fechas: convertir a string YYYY-MM-DD.
    # python-frontmatter puede parsear las fechas como datetime.date,
    # así que las convertimos a string para consistencia.
    created = _date_to_str(meta.get("created"))
    updated = _date_to_str(meta.get("updated"))

    # Campos extra: todo lo que no sea core.
    core_keys = {"title", "tags", "created", "updated"}
    extra_fields = {k: v for k, v in meta.items() if k not in core_keys}

    # Ruta relativa como identificador de la nota.
    source_path = filepath.name

    metadata = NoteMetadata(
        source_path=source_path,
        title=title,
        tags=tags,
        note_type=note_type,
        created=created,
        updated=updated,
        extra_fields=extra_fields,
    )

    # --- Limpiar el body ---
    clean_body = _clean_wikilinks(body)

    logger.debug(
        "Nota parseada: %s (tipo=%s, tags=%d, body=%d chars)",
        source_path, note_type, len(tags), len(clean_body),
    )

    return ParsedNote(metadata=metadata, body=clean_body)


def _date_to_str(value) -> str | None:
    """
    Convierte un valor de fecha del frontmatter a string YYYY-MM-DD.

    python-frontmatter puede devolver:
    - datetime.date → lo convertimos con str()
    - string "2026-09-15" → lo dejamos tal cual
    - None → None
    - Otro tipo → str() como fallback
    """
    if value is None:
        return None
    return str(value)


# =============================================================================
# ESCANEO — Búsqueda recursiva de ficheros Markdown
# =============================================================================

def scan_notes(directory: str | Path) -> list[Path]:
    """
    Escanea recursivamente un directorio buscando ficheros .md.

    Ignora directorios según config.NOTES_IGNORE_PATTERNS. Cada patrón puede
    ser un nombre exacto (ej: "4-meta") o un glob estilo fnmatch (ej: ".*"
    para ignorar todas las carpetas ocultas como .obsidian, .app, .vscode).
    Esto evita indexar configuraciones de Obsidian, ficheros borrados,
    y plantillas.

    Args:
        directory: Ruta al directorio raíz de notas.

    Returns:
        Lista ordenada de Paths a ficheros .md encontrados.
        Ordenada para que la indexación sea determinista (mismo orden siempre).
    """
    directory = Path(directory)

    if not directory.is_dir():
        logger.warning("Directorio de notas no encontrado: %s", directory)
        return []

    ignore = config.NOTES_IGNORE_PATTERNS
    paths: list[Path] = []

    for path in sorted(directory.rglob("*.md")):
        # Verificar que ningún componente del path coincide con los patrones
        # de ignorados. Soporta nombres exactos ("4-meta") y globs (".*").
        # Ejemplo: notas/.obsidian/workspace.md → ".obsidian" coincide con ".*".
        parts = path.relative_to(directory).parts[:-1]  # directorios, sin el fichero
        if _is_ignored(parts, ignore):
            logger.debug("Ignorando (directorio excluido): %s", path)
            continue
        paths.append(path)

    logger.info("Encontrados %d ficheros .md en %s", len(paths), directory)
    return paths


def _is_ignored(parts: tuple | list, patterns: list[str]) -> bool:
    """
    Comprueba si algún componente del path coincide con los patrones.

    Usa fnmatch para que ".*" ignore todas las carpetas ocultas
    (.obsidian, .app, .vscode, .stversions...) presentes y futuras.
    Sin comodines, fnmatch equivale a igualdad exacta.
    """
    return any(
        fnmatch.fnmatch(part, pat) for part in parts for pat in patterns
    )
