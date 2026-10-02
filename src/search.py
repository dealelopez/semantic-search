# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# search.py — Motor de búsqueda semántica y formateo de resultados
# =============================================================================
#
# Este módulo conecta la query del usuario con los vectores almacenados
# en ChromaDB. El flujo es:
#
#   1. Usuario escribe "cómo funcionan las redes neuronales"
#   2. embed_query() convierte la query en un vector de 768 dimensiones
#   3. store.search() busca los vectores más cercanos en ChromaDB
#   4. format_results() muestra los resultados bonitos en terminal
#
# El mismo modelo para indexar y buscar:
# ---
# Es CRÍTICO que el embedding de la query se genere con el mismo modelo
# que los embeddings de los documentos. Los vectores de modelos diferentes
# viven en espacios matemáticos distintos y no son comparables. Es como
# intentar medir distancias mezclando kilómetros y millas — los números
# no significan nada cuando se comparan.
#
# El "semantic gap":
# ---
# A veces los resultados sorprenden. Si buscas "vacaciones" podrías
# encontrar un texto sobre "días de descanso laboral" aunque no use la
# palabra "vacaciones". Eso es el poder del embedding — captura significado,
# no palabras. Pero también puede fallar: si el concepto no está bien
# representado en los datos de entrenamiento del modelo, el matching puede
# ser impreciso. Por eso los scores de similitud son útiles para juzgar
# la confianza del resultado.
# =============================================================================

import json
import logging
import time
from dataclasses import asdict

from rich.console import Console

from src import config
from src.embeddings import OllamaEmbedder
from src.models import SearchResult
from src.store import VectorStore

logger = logging.getLogger(__name__)

console = Console()


class SearchEngine:
    """
    Motor de búsqueda semántica.

    Recibe un embedder (genera vectores de queries) y un store (busca en
    ChromaDB). Coordina la búsqueda y formatea los resultados para terminal.
    """

    def __init__(self, embedder: OllamaEmbedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    # =========================================================================
    # BÚSQUEDA
    # =========================================================================

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_tags: list[str] | None = None,
        filter_type: str | None = None,
        keyword: str | None = None,
        min_score: float = 0.0,
    ) -> tuple[list[SearchResult], float]:
        """
        Busca notas relevantes para una query en lenguaje natural.

        Args:
            query: Texto de la consulta.
            n_results: Número de resultados a devolver.
            filter_tags: Filtrar por tags (solo chunks que contengan estos tags).
            filter_type: Filtrar por tipo de nota (nota, proyecto, receta...).
            keyword: Término exacto que debe aparecer en el texto (búsqueda híbrida).
            min_score: Umbral mínimo de similitud (0.0-1.0). Descarta resultados por debajo.

        Returns:
            Tupla (lista de SearchResult, elapsed en segundos).
        """
        start = time.time()

        # 1. Generar embedding de la query.
        query_embedding = self.embedder.embed_query(query)

        # 2. Construir filtros ChromaDB.
        where_filter = _build_where_filter(filter_type, filter_tags)
        where_document = {"$contains": keyword} if keyword else None

        # 3. Buscar en ChromaDB.
        # Pedir más resultados si hay min_score para compensar los filtrados.
        fetch_n = n_results * 3 if min_score > 0 else n_results
        results = self.store.search(
            query_embedding=query_embedding,
            n_results=fetch_n,
            where_filter=where_filter,
            where_document=where_document,
        )

        # 4. Filtrar por score mínimo.
        effective_min = min_score or config.MIN_SCORE
        if effective_min > 0:
            results = [r for r in results if r.score >= effective_min]

        # 5. Limitar a n_results.
        results = results[:n_results]

        elapsed = time.time() - start
        logger.debug(
            "Búsqueda '%s': %d resultados en %.2fs",
            query, len(results), elapsed,
        )
        return results, elapsed

    # =========================================================================
    # FORMATEO DE RESULTADOS
    # =========================================================================

    def format_results(
        self,
        query: str,
        results: list[SearchResult],
        elapsed: float,
        max_preview: int | None = None,
    ) -> None:
        """
        Formatea e imprime los resultados usando rich para output bonito.

        Args:
            max_preview: Longitud máxima del preview. None = config default.
                         Pasar un valor grande para --full.
        """
        preview_len = max_preview or config.PREVIEW_MAX_CHARS
        console.print(f"\n[bold]Resultados para: \"{query}\"[/bold]")
        console.print("━" * 55)

        if not results:
            console.print("[dim]No se encontraron resultados.[/dim]")
            console.print()
            console.print("[dim]Sugerencias:[/dim]")
            console.print("[dim]   - ¿Has indexado las notas? Ejecuta: index --mode full[/dim]")
            console.print("[dim]   - Prueba con otros términos o sinónimos[/dim]")
            console.print("[dim]   - Si usas filtros, prueba sin ellos[/dim]")
            console.print("━" * 55)
            console.print(f"0 resultados en {elapsed:.2f}s")
            return

        for i, r in enumerate(results, 1):
            # Línea principal: número + fichero + heading
            header = f" {i}. {r.source}"
            if r.heading:
                header += f" > {r.heading}"
            console.print(f"[bold]{header}[/bold]")

            # Score con color según valor.
            # Verde (≥0.8): muy relevante, el resultado probablemente es lo que buscas.
            # Amarillo (0.6-0.8): relevante, pero podría no ser exactamente lo que quieres.
            # Rojo (<0.6): baja relevancia, revisar con cuidado.
            color = "green" if r.score >= 0.8 else "yellow" if r.score >= 0.6 else "red"
            console.print(f"    Similitud: [{color}]{r.score:.2f}[/{color}]")

            # Tags.
            if r.tags:
                tags_str = " · ".join(r.tags)
                console.print(f"    Tags: {tags_str}")

            # Preview del contenido (truncado a preview_len).
            preview = r.text[:preview_len]
            if len(r.text) > preview_len:
                preview += "..."
            # Indentar y limpiar saltos de línea para que quede legible.
            preview_lines = preview.replace("\n", "\n       ")
            console.print(f"    {preview_lines}")
            console.print()  # Línea en blanco entre resultados

        console.print("━" * 55)
        console.print(f"{len(results)} resultado(s) en {elapsed:.2f}s")

    def format_results_json(
        self,
        results: list[SearchResult],
        elapsed: float,
    ) -> None:
        """Imprime los resultados en formato JSON (para scripting/pipes)."""
        output = {
            "results": [asdict(r) for r in results],
            "total": len(results),
            "elapsed_seconds": round(elapsed, 3),
        }
        # ensure_ascii=False para caracteres españoles (ñ, acentos, emojis).
        print(json.dumps(output, ensure_ascii=False, indent=2))

    def format_results_explain(
        self,
        query: str,
        results: list[SearchResult],
        elapsed: float,
        filter_type: str | None = None,
        filter_tags: list[str] | None = None,
    ) -> None:
        """Muestra resultados con información de debug detallada."""
        console.print("\n[bold]Modo explain[/bold]")
        console.print("━" * 55)

        # Info de la colección.
        try:
            stats = self.store.collection_stats()
            console.print(f"   Chunks en colección: {stats['total_chunks']:,}")
            console.print(f"   Notas indexadas:     {stats['total_sources']}")
        except Exception:
            console.print("   [dim]No se pudo obtener stats de la colección[/dim]")

        # Filtro usado.
        where = _build_where_filter(filter_type, filter_tags)
        if where:
            console.print(f"   Filtro ChromaDB:     {where}")
        else:
            console.print("   Filtro ChromaDB:     ninguno")

        console.print(f"   Modelo embeddings:   {self.embedder.model}")
        console.print(f"   Tiempo total:        {elapsed:.3f}s")
        console.print("━" * 55)

        if not results:
            console.print("[dim]Sin resultados.[/dim]")
            return

        for i, r in enumerate(results, 1):
            distance = round(1.0 - r.score, 4)
            header = f" {i}. {r.source}"
            if r.heading:
                header += f" > {r.heading}"
            console.print(f"[bold]{header}[/bold]")
            console.print(
                f"    Score: {r.score:.4f}  |  Distancia coseno: {distance:.4f}"
            )
            console.print(f"    Tipo: {r.note_type}  |  Tags: {', '.join(r.tags)}")
            if r.created:
                console.print(f"    Creado: {r.created}")
            preview = r.text[:150].replace("\n", " ")
            console.print(f"    Preview: {preview}...")
            console.print()

        console.print("━" * 55)
        console.print(f"{len(results)} resultado(s) en {elapsed:.3f}s")


# =============================================================================
# CONSTRUCCIÓN DE FILTROS — where clause para ChromaDB
# =============================================================================
# ChromaDB soporta filtros sobre metadata. Primero filtra por metadata
# (operación rápida sobre un índice) y luego busca por similitud solo
# entre los resultados filtrados. Más eficiente que buscar todos y
# filtrar después.
#
# En ChromaDB 1.5+, los tags se almacenan como arrays nativos y se
# filtran con $contains. No hace falta parsear strings CSV.
# =============================================================================

def _build_where_filter(
    filter_type: str | None,
    filter_tags: list[str] | None,
) -> dict | None:
    """
    Construye el filtro where para ChromaDB a partir de los filtros del usuario.

    Ejemplos:
        filter_type="nota"           → {"note_type": {"$eq": "nota"}}
        filter_tags=["ia"]           → {"tags": {"$contains": "ia"}}
        filter_tags=["ia", "ml"]     → {"$and": [{"tags": {"$contains": "ia"}},
                                                  {"tags": {"$contains": "ml"}}]}
        tipo + tags                  → {"$and": [{tipo}, {tags...}]}
    """
    conditions: list[dict] = []

    if filter_type:
        conditions.append({"note_type": {"$eq": filter_type}})

    if filter_tags:
        for tag in filter_tags:
            conditions.append({"tags": {"$contains": tag}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# =============================================================================
# DEDUPLICACIÓN — Un resultado por nota
# =============================================================================

def deduplicate_by_source(results: list[SearchResult]) -> list[SearchResult]:
    """
    Elimina resultados duplicados por fichero fuente, dejando el de mayor score.

    Útil cuando múltiples chunks de la misma nota coinciden con la query.
    Con --unique en la CLI, el usuario ve un resultado por nota.
    """
    seen: dict[str, SearchResult] = {}
    for r in results:
        if r.source not in seen or r.score > seen[r.source].score:
            seen[r.source] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)
