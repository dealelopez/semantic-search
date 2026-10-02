# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# models.py — Estructuras de datos del buscador semántico
# =============================================================================
#
# Todas las dataclasses que se comparten entre módulos. Centralizar los modelos
# aquí evita imports circulares y hace que la arquitectura sea más clara:
# cualquier módulo que necesite una estructura de datos la importa de aquí.
#
# Usamos dataclasses en lugar de dicts porque:
# 1. Autocompletado en el editor (el IDE sabe qué campos tiene cada objeto)
# 2. Errores tempranos (si olvidas un campo, Python explota al crear el objeto)
# 3. Documentación implícita (los tipos dicen qué espera cada campo)
# 4. Inmutabilidad opcional (frozen=True si se necesita)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
# PARSING — Resultado de leer y parsear un fichero Markdown
# =============================================================================

@dataclass
class NoteMetadata:
    """
    Metadatos extraídos del frontmatter YAML de una nota Markdown.

    En el sistema de notas del usuario, el frontmatter sigue un formato "v2"
    donde el primer tag indica el tipo de nota (nota, proyecto, receta, etc.).
    Esto permite filtrar resultados de búsqueda por categoría.

    Ejemplo de frontmatter:
        ---
        title: Embeddings y búsqueda semántica
        tags:
          - nota
          - inteligencia-artificial
        created: 2026-09-15
        ---

    Si una nota no tiene frontmatter (el ~2% del corpus), se generan
    metadatos por defecto a partir del nombre del fichero.
    """

    source_path: str
    """Ruta relativa del fichero (ej: 'proyecto-x.md'). Sirve como identificador único."""

    title: str
    """Título legible. Viene del campo 'title' del YAML, o del nombre del fichero sin extensión."""

    tags: list[str] = field(default_factory=list)
    """Lista de tags (ej: ['nota', 'ia', 'estado/seedling']). Vacía si no hay frontmatter."""

    note_type: str = "sin-tipo"
    """
    Tipo de nota, derivado del primer tag (sistema v2).
    Ejemplos: 'nota', 'proyecto', 'receta', 'reunion'.
    'sin-tipo' si no hay tags.
    """

    created: str | None = None
    """Fecha de creación (YYYY-MM-DD) o None si no se especifica."""

    updated: str | None = None
    """Última modificación o None."""

    extra_fields: dict = field(default_factory=dict)
    """
    Campos adicionales del frontmatter que no son los "core" (title, tags, created, updated).
    Ejemplos: status, maker, url, rating, people, org...
    Se guardan por si algún módulo los necesita después.
    """

    content_hash: str | None = None
    """Hash MD5 del contenido del fichero. Se usa en indexación incremental para detectar cambios."""


@dataclass
class ParsedNote:
    """
    Resultado completo de parsear un fichero Markdown.

    Contiene los metadatos (del frontmatter) y el cuerpo limpio del documento
    (sin frontmatter, sin wikilinks de Obsidian).
    """

    metadata: NoteMetadata
    """Metadatos extraídos del frontmatter YAML."""

    body: str
    """
    Cuerpo markdown limpio (sin el bloque YAML, sin wikilinks).
    Es el texto que se trocea y se convierte en embeddings.
    """


# =============================================================================
# CHUNKING — Fragmentos de texto para generar embeddings
# =============================================================================

@dataclass
class Chunk:
    """
    Un fragmento de texto listo para convertir en embedding.

    El chunking divide documentos largos en fragmentos más pequeños porque
    un embedding de un texto corto y enfocado (100-300 tokens) captura mejor
    el significado que un embedding de un documento entero. Es la diferencia
    entre un vector que dice "esto habla de Redis" y uno que dice "esto habla
    de muchas cosas, incluyendo Redis, Docker, y recetas de cocina".

    Cada chunk registra su posición y contexto dentro del documento original
    para poder mostrar en los resultados dónde exactamente se encontró el match.
    """

    text: str
    """Contenido textual del chunk (lo que se convierte en embedding)."""

    heading_path: str
    """
    Ruta de headings bajo la que cae este chunk.
    Ejemplo: '## Configuración > ### Redis'
    Se usa para mostrar en los resultados dónde está el fragmento
    dentro del documento original. Vacío si no hay headings.
    """

    chunk_index: int
    """Posición del chunk dentro de la nota (0, 1, 2...). Para construir IDs deterministas."""

    token_count: int
    """Tokens estimados del chunk (~len(words) * 1.3 para español)."""

    source_path: str
    """Ruta del fichero origen. Necesario para vincular el chunk con su nota."""


# =============================================================================
# BÚSQUEDA — Resultado de una búsqueda semántica
# =============================================================================

@dataclass
class SearchResult:
    """
    Un resultado individual de búsqueda semántica.

    ChromaDB devuelve "distancia coseno" (menor = más similar), pero nosotros
    la convertimos a "score de similitud" (mayor = más similar) porque es
    más intuitivo: 0.95 = muy relevante, 0.30 = poco relevante.
    """

    source: str
    """Ruta del fichero origen (ej: 'proyecto-x.md')."""

    heading: str
    """Heading del chunk (ej: '## Configuración > ### Redis')."""

    text: str
    """Texto completo del chunk encontrado."""

    score: float
    """
    Similitud (0.0 a 1.0, mayor = más similar).
    Calculado como 1 - distancia_coseno.
    """

    note_type: str
    """Tipo de nota (primer tag del frontmatter)."""

    tags: list[str] = field(default_factory=list)
    """Lista de tags de la nota."""

    created: str | None = None
    """Fecha de creación de la nota."""


# =============================================================================
# INDEXACIÓN — Reporte de una operación de indexación
# =============================================================================

@dataclass
class IndexReport:
    """
    Reporte detallado de una operación de indexación (full o incremental).

    Se devuelve al finalizar la indexación para que la CLI pueda mostrar
    un resumen al usuario: cuántas notas se procesaron, cuántos chunks
    se crearon, si hubo errores, y cuánto tardó.
    """

    notes_processed: int = 0
    """Notas procesadas (nuevas + modificadas)."""

    notes_skipped: int = 0
    """Notas sin cambios (solo relevante en modo incremental)."""

    notes_deleted: int = 0
    """Notas eliminadas del índice (solo relevante en modo incremental)."""

    chunks_created: int = 0
    """Total de chunks generados e insertados en ChromaDB."""

    errors: list[str] = field(default_factory=list)
    """
    Lista de errores (fichero + mensaje). Cada error es un string descriptivo.
    El indexer continúa procesando las demás notas aunque una falle.
    """

    duration_seconds: float = 0.0
    """Tiempo total de ejecución en segundos."""
