# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# chunker.py — Troceado inteligente de texto para embeddings
# =============================================================================
#
# El chunking es probablemente la decisión más importante en un sistema de
# búsqueda semántica. Un embedding (vector de 768 números) debe capturar el
# "significado" del texto. Si el texto es muy largo (un documento entero),
# el vector resultante es demasiado difuso — intenta representar demasiados
# conceptos y no captura bien ninguno. Si es muy corto (una frase suelta),
# pierde contexto y no tiene suficiente información semántica.
#
# El punto dulce para nomic-embed-text está entre 100-300 tokens (~70-220
# palabras en español). Un chunk de ese tamaño suele cubrir una idea completa:
# un párrafo explicativo, una sección de configuración, una receta, etc.
#
# Estrategia híbrida:
# ---
# En lugar de usar una sola técnica de chunking, combinamos tres niveles:
#
# 1. Headings (##, ###) — Respeta la estructura del documento. Si el autor
#    organizó su texto con secciones, cada sección es probablemente una unidad
#    temática coherente. Es el mejor corte posible.
#
# 2. Párrafos — Fallback si no hay headings o si una sección es muy larga.
#    Un párrafo separado por línea en blanco suele ser una unidad de sentido.
#
# 3. Tamaño fijo — Último recurso si un párrafo es enorme (ej: un log pegado
#    sin formatear). Corta mecánicamente con overlap para no perder ideas
#    que caigan en el borde.
#
# ¿Por qué no solo tamaño fijo?
# Porque un corte mecánico puede partir una idea por la mitad. Si una sección
# de 250 tokens explica cómo configurar Redis, un corte a 150 tokens perdería
# el contexto. Respetar headings y párrafos produce chunks más coherentes.
#
# Clase abstracta (Open/Closed Principle):
# ---
# Definimos ChunkingStrategy como clase abstracta para que sea fácil crear
# otras estrategias (FixedSizeChunker, SemanticChunker, etc.) sin cambiar
# el código que usa chunkers (Indexer, tests, etc.).
# =============================================================================

import logging
import re
from abc import ABC, abstractmethod

from src import config
from src.models import Chunk, NoteMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# ESTIMACIÓN DE TOKENS
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Aproximación rápida del número de tokens en un texto.

    Fórmula: número de palabras × 1.3

    ¿Por qué 1.3?
    Un "token" en modelos de lenguaje no es exactamente una palabra.
    Los tokenizers (como el de nomic-embed-text) dividen las palabras
    en sub-unidades. En inglés, la media es ~1.3 tokens por palabra.
    En español, con palabras más largas y acentos, puede ser ~1.4.
    Usamos 1.3 como compromiso razonable.

    ¿Por qué no usar un tokenizer real (como tiktoken)?
    Para chunking, una aproximación del ±20% es suficiente. Lo que importa
    es que los chunks estén "más o menos" entre 100 y 300 tokens, no que
    sean exactamente 247. Un tokenizer real añadiría una dependencia pesada
    y sería más lento, sin mejorar significativamente los resultados.
    La precisión exacta importa cuando pagas por token (APIs de OpenAI),
    no cuando controlas el modelo localmente.
    """
    words = text.split()
    if not words:
        return 0
    return int(len(words) * 1.3)


# =============================================================================
# CLASE ABSTRACTA — Interfaz de chunking
# =============================================================================

class ChunkingStrategy(ABC):
    """
    Interfaz abstracta para estrategias de chunking.

    Permite crear diferentes estrategias (por headings, por tamaño fijo,
    semántico, etc.) sin cambiar el código que las consume. El Indexer
    recibe un ChunkingStrategy y no sabe ni le importa cuál es la
    implementación concreta.

    Principio Open/Closed: abierto a extensión (nuevas estrategias),
    cerrado a modificación (no se cambia el Indexer para soportarlas).
    """

    @abstractmethod
    def chunk(self, body: str, metadata: NoteMetadata) -> list[Chunk]:
        """
        Divide el body de una nota en chunks listos para generar embeddings.

        Args:
            body: Texto markdown limpio (sin frontmatter, sin wikilinks).
            metadata: Metadatos de la nota (para source_path en cada chunk).

        Returns:
            Lista de Chunks. Puede estar vacía si el body está vacío.
        """
        ...


# =============================================================================
# HYBRID CHUNKER — Implementación principal
# =============================================================================

class HybridChunker(ChunkingStrategy):
    """
    Estrategia híbrida: headings → párrafos → tamaño fijo.

    Algoritmo:
    1. Si el documento tiene headings (##, ###), divide por headings.
       Cada sección se convierte en un chunk (o varios si es muy larga).
    2. Si no tiene headings, divide por párrafos (líneas en blanco).
       Agrupa párrafos cortos consecutivos hasta alcanzar MIN_CHUNK_TOKENS.
    3. Si un fragmento (sección o párrafo) supera MAX_CHUNK_TOKENS,
       lo subdivide por tamaño fijo con OVERLAP_TOKENS de solapamiento.

    Parámetros configurables vía config.py / variables de entorno:
    - MAX_CHUNK_TOKENS (default 300): tamaño máximo de un chunk
    - MIN_CHUNK_TOKENS (default 100): tamaño mínimo (se agrupan los pequeños)
    - OVERLAP_TOKENS (default 50): solapamiento entre chunks al cortar por tamaño
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ):
        self.max_tokens = max_tokens or config.MAX_CHUNK_TOKENS
        self.min_tokens = min_tokens or config.MIN_CHUNK_TOKENS
        self.overlap_tokens = overlap_tokens or config.OVERLAP_TOKENS

    def chunk(self, body: str, metadata: NoteMetadata) -> list[Chunk]:
        """Divide el body en chunks usando la estrategia híbrida."""

        # Body vacío → lista vacía (no explota).
        if not body or not body.strip():
            return []

        source = metadata.source_path

        # Intentar dividir por headings primero.
        sections = _split_by_headings(body)

        if sections:
            # El documento tiene headings → procesar por secciones.
            raw_chunks = self._chunks_from_sections(sections)
        else:
            # Sin headings → procesar por párrafos.
            raw_chunks = self._chunks_from_paragraphs(body)

        # Construir objetos Chunk con índice secuencial.
        chunks: list[Chunk] = []
        for i, (text, heading_path) in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            chunks.append(Chunk(
                text=text,
                heading_path=heading_path,
                chunk_index=i,
                token_count=estimate_tokens(text),
                source_path=source,
            ))

        logger.debug(
            "Chunked %s: %d chunks (body %d chars)",
            source, len(chunks), len(body),
        )
        return chunks

    # ----- Procesamiento por secciones (con headings) -----

    def _chunks_from_sections(
        self,
        sections: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Genera chunks a partir de secciones divididas por headings.

        Cada sección se evalúa: si cabe en MAX_CHUNK_TOKENS, se queda como
        chunk entero. Si es demasiado larga, se subdivide por párrafos
        y, si es necesario, por tamaño fijo.

        Returns:
            Lista de (texto, heading_path) para cada chunk.
        """
        result: list[tuple[str, str]] = []

        for heading_path, content in sections:
            content = content.strip()
            if not content:
                continue

            tokens = estimate_tokens(content)

            if tokens <= self.max_tokens:
                # La sección entera cabe en un chunk.
                result.append((content, heading_path))
            else:
                # Sección demasiado larga → subdividir por párrafos.
                paragraphs = _split_by_paragraphs(content)
                sub_chunks = self._process_paragraph_list(paragraphs)
                for sub_text in sub_chunks:
                    result.append((sub_text, heading_path))

        return result

    # ----- Procesamiento por párrafos (sin headings) -----

    def _chunks_from_paragraphs(self, body: str) -> list[tuple[str, str]]:
        """
        Genera chunks a partir de párrafos cuando no hay headings.

        Delega en _process_paragraph_list que agrupa párrafos cortos
        y subdivide los grandes. No pre-agrupa para evitar doble agrupación.

        Returns:
            Lista de (texto, heading_path) — heading_path será vacío.
        """
        paragraphs = _split_by_paragraphs(body)
        sub_chunks = self._process_paragraph_list(paragraphs)
        return [(text, "") for text in sub_chunks]

    # ----- Subdivisión de párrafos por tamaño -----

    def _process_paragraph_list(self, paragraphs: list[str]) -> list[str]:
        """
        Procesa una lista de párrafos: agrupa los pequeños, subdivide los grandes.
        """
        grouped = _group_small_paragraphs(paragraphs, self.min_tokens)
        result: list[str] = []

        for group in grouped:
            if estimate_tokens(group) > self.max_tokens:
                # Demasiado largo → corte mecánico con overlap.
                result.extend(
                    _split_by_size(group, self.max_tokens, self.overlap_tokens)
                )
            else:
                result.append(group)

        return result


# =============================================================================
# FUNCIONES AUXILIARES — Dividir y agrupar texto
# =============================================================================

# Regex para detectar líneas de heading markdown (nivel 1-6).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_by_headings(body: str) -> list[tuple[str, str]]:
    """
    Divide el body por líneas de heading (##, ###, etc.).

    Cada sección incluye el heading que la inicia y todo el texto hasta
    el siguiente heading del mismo o menor nivel.

    Maneja headings anidados construyendo un "heading_path":
      ## Configuración > ### Redis

    Args:
        body: Texto markdown.

    Returns:
        Lista de (heading_path, content).
        Lista vacía si no hay headings.
    """
    matches = list(_HEADING_RE.finditer(body))

    if not matches:
        return []

    sections: list[tuple[str, str]] = []

    # Si hay texto antes del primer heading, incluirlo como sección sin heading.
    first_pos = matches[0].start()
    preamble = body[:first_pos].strip()
    if preamble:
        sections.append(("", preamble))

    # Pila de headings para construir heading_path.
    # Cada entrada: (nivel, texto_heading_completo)
    heading_stack: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))  # número de #
        heading_text = match.group(0).strip()  # ej: "## Configuración"

        # Actualizar la pila: eliminar headings de nivel >= al actual.
        # Si estamos en ### y aparece ##, eliminamos el ### anterior.
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))

        # Construir heading_path concatenando la pila.
        heading_path = " > ".join(h[1] for h in heading_stack)

        # Extraer el contenido entre este heading y el siguiente.
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()

        sections.append((heading_path, content))

    return sections


def _split_by_paragraphs(text: str) -> list[str]:
    """
    Divide texto por líneas en blanco (separador de párrafos en Markdown).

    Filtra párrafos vacíos y strips whitespace.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _group_small_paragraphs(paragraphs: list[str], min_tokens: int) -> list[str]:
    """
    Agrupa párrafos consecutivos hasta alcanzar min_tokens.

    Si un párrafo individual ya supera min_tokens, no se agrupa con nadie.
    Si varios párrafos consecutivos son cortos, se unen con doble salto
    de línea hasta alcanzar el mínimo.

    Esto evita chunks tan cortos que no tienen suficiente información
    semántica para un embedding útil. Una frase suelta como "Esto funciona
    bien" no produce un vector significativo. Con contexto alrededor, sí.
    """
    if not paragraphs:
        return []

    groups: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens >= min_tokens and current_parts:
            # El grupo actual ya alcanza el mínimo → cerrar y empezar nuevo.
            groups.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = para_tokens
        else:
            # Seguir acumulando.
            current_parts.append(para)
            current_tokens += para_tokens

    # Último grupo pendiente.
    if current_parts:
        # Si el último grupo es muy corto y hay grupos anteriores,
        # anexarlo al último grupo.
        if groups and current_tokens < min_tokens:
            groups[-1] = groups[-1] + "\n\n" + "\n\n".join(current_parts)
        else:
            groups.append("\n\n".join(current_parts))

    return groups


def _split_by_size(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Corte mecánico por tamaño fijo con solapamiento (overlap).

    ¿Qué es el overlap?
    Imagina que una idea clave cae justo en el borde entre dos chunks.
    Sin overlap, esa idea se parte en dos y ninguno la captura completa.
    Con overlap, los últimos ~50 tokens del chunk N se repiten al inicio
    del chunk N+1, asegurando que la idea aparece completa en al menos uno.

    El trade-off: más overlap = mejor cobertura semántica pero más chunks
    (y por tanto más embeddings que generar y almacenar).

    Args:
        text: Texto a dividir.
        max_tokens: Tokens máximos por chunk.
        overlap_tokens: Tokens de solapamiento entre chunks consecutivos.

    Returns:
        Lista de fragmentos de texto.
    """
    words = text.split()
    if not words:
        return []

    # Convertir tokens a palabras (inverso de estimate_tokens).
    max_words = max(1, int(max_tokens / 1.3))
    overlap_words = max(0, int(overlap_tokens / 1.3))

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)

        if end >= len(words):
            break

        # Avanzar, retrocediendo el overlap.
        start = end - overlap_words
        # Evitar bucle infinito si overlap >= max_words.
        if start <= (end - max_words):
            start = end

    return chunks
