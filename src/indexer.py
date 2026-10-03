# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# indexer.py — Orquestación de indexación completa e incremental
# =============================================================================
#
# El Indexer es el "director de orquesta" del pipeline de indexación.
# No hace nada por sí mismo — delega en los módulos especializados:
#
#   1. scan_notes()     → lista de ficheros .md en disco
#   2. parse_note()     → extrae frontmatter + body limpio
#   3. chunker.chunk()  → trocea el body en fragmentos
#   4. embedder.embed() → convierte cada fragmento en un vector
#   5. store.upsert()   → almacena vectores + metadata en ChromaDB
#
# Dos modos de operación:
# ---
# - Full reindex: borra todo y reprocesa desde cero. Simple, seguro,
#   siempre correcto. Lento (~3-8 min para 400 notas en CPU).
#   Cuándo usar: primera indexación, cambio de modelo, cambio de chunking.
#
# - Incremental: solo procesa lo que cambió (nuevo, modificado, eliminado).
#   Rápido (segundos si solo cambió una nota). Más complejo porque necesita
#   detectar cambios comparando hashes del contenido.
#   Cuándo usar: día a día, tras añadir/editar unas pocas notas.
#
# Inyección de dependencias:
# ---
# El Indexer recibe embedder, store y chunker en el constructor. No los
# crea por sí mismo. Esto permite:
#   1. Testear con mocks (sin Ollama ni Docker)
#   2. Cambiar implementaciones (ej: Qdrant en vez de ChromaDB)
#   3. Cada componente es independiente y testeable por separado
#
# Resiliencia a errores:
# ---
# Si una nota falla (frontmatter malformado, timeout de Ollama, etc.),
# el indexer la salta y continúa con la siguiente. Los errores se acumulan
# en el IndexReport para revisarlos después. No queremos que una nota
# problemática pare la indexación de las otras 399.
#
# Idempotencia:
# ---
# Puedes ejecutar full_reindex o incremental_index las veces que quieras.
# Full borra y recrea. Incremental usa upsert (si el ID existe, se
# sobreescribe). No se crean duplicados nunca.
# =============================================================================

import hashlib
import logging
import time
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from src.chunker import ChunkingStrategy
from src.embeddings import OllamaEmbedder
from src.frontmatter_parser import parse_note, scan_notes
from src.models import IndexReport
from src.store import VectorStore

logger = logging.getLogger(__name__)


class Indexer:
    """
    Orquesta el pipeline de indexación: leer → parsear → trocear → embeder → almacenar.

    Recibe las dependencias inyectadas (embedder, store, chunker) y las coordina.
    No crea nada por sí mismo — solo orquesta.
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        store: VectorStore,
        chunker: ChunkingStrategy,
    ):
        self.embedder = embedder
        self.store = store
        self.chunker = chunker

    # =========================================================================
    # FULL REINDEX — Borra todo y reprocesa desde cero
    # =========================================================================

    def full_reindex(self, notes_dir: str, show_progress: bool = True) -> IndexReport:
        """
        Reindexación completa en dos fases: parse+chunk → embed batch → upsert.

        Fase 1: Parsea y trocea todas las notas (rápido, CPU-only).
        Fase 2: Genera todos los embeddings en un solo embed_documents() call.
                 Esto optimiza el batching: en vez de una petición HTTP por nota
                 (~5 chunks cada una), envía todos los chunks juntos y el
                 batching interno de embed_documents() los agrupa en lotes
                 óptimos de EMBED_BATCH_SIZE. Para 400 notas con ~2000 chunks,
                 esto reduce de ~400 a ~40 peticiones HTTP.
        Fase 3: Upsert por nota en ChromaDB (rápido, local).

        Args:
            notes_dir: Ruta al directorio de notas.
            show_progress: Mostrar barra de progreso (False en tests).

        Returns:
            IndexReport con totales y duración.
        """
        start = time.time()
        report = IndexReport()

        # 1. Borrar la colección entera.
        self.store.reset_collection()
        logger.info("Colección reseteada — indexación completa iniciada")

        # 2. Escanear notas.
        paths = scan_notes(notes_dir)
        if not paths:
            report.duration_seconds = time.time() - start
            return report

        # ── Fase 1: Parse + chunk (rápido) ──
        prepared: list[tuple] = []  # (parsed, chunks)
        iterator = _progress_iterator(paths, "Parseando notas...", show_progress)

        for path in iterator:
            try:
                parsed = parse_note(path)
                chunks = self.chunker.chunk(parsed.body, parsed.metadata)
                report.notes_processed += 1
                if chunks:
                    parsed.metadata.content_hash = compute_content_hash(path)
                    prepared.append((parsed, chunks))
            except Exception as e:
                report.errors.append(f"{path.name}: {e}")
                logger.warning("Error parseando %s: %s", path.name, e)

        if not prepared:
            report.duration_seconds = time.time() - start
            return report

        # ── Fase 2: Embed batch global (lento, I/O) ──
        all_texts: list[str] = []
        for parsed, chunks in prepared:
            title = parsed.metadata.title
            for c in chunks:
                all_texts.append(f"{title}\n\n{c.text}")

        try:
            all_embeddings = self.embedder.embed_documents(all_texts)
        except Exception as e:
            report.errors.append(f"Error generando embeddings: {e}")
            report.duration_seconds = time.time() - start
            return report

        # ── Fase 3: Upsert por nota (rápido) ──
        # Cada upsert va protegido: si una nota falla (ej: metadata que
        # ChromaDB rechaza), se registra el error y se continúa con las
        # demás en vez de abortar toda la indexación y perder el trabajo
        # de embeddings ya hecho.
        offset = 0
        for parsed, chunks in prepared:
            n = len(chunks)
            note_embeddings = all_embeddings[offset : offset + n]
            try:
                self.store.upsert_chunks(chunks, note_embeddings, parsed.metadata)
                report.chunks_created += n
            except Exception as e:
                error_msg = f"{parsed.metadata.source_path}: {e}"
                report.errors.append(error_msg)
                logger.warning(
                    "Error guardando %s: %s", parsed.metadata.source_path, e
                )
            offset += n

        report.duration_seconds = time.time() - start
        logger.info(
            "Indexación completa: %d notas, %d chunks, %d errores en %.1fs",
            report.notes_processed, report.chunks_created,
            len(report.errors), report.duration_seconds,
        )
        return report

    # =========================================================================
    # INCREMENTAL INDEX — Solo cambios (nuevo/modificado/eliminado)
    # =========================================================================

    def incremental_index(self, notes_dir: str, show_progress: bool = True) -> IndexReport:
        """
        Indexación incremental: solo procesa lo que cambió desde la última vez.

        Compara hashes MD5 del contenido de cada fichero con los almacenados
        en ChromaDB. Solo re-procesa notas nuevas o modificadas, y elimina
        del índice las notas que ya no están en disco.

        Por qué hash del contenido y no fecha de modificación:
        El mtime del fichero puede cambiar sin que el contenido cambie
        (ej: rsync, git checkout, sync de Obsidian). El hash es la única
        forma fiable de detectar cambios reales.

        Cuándo usar:
        - Después de añadir o editar unas pocas notas
        - Como comando habitual del día a día
        - NO es válido si cambiaste el modelo o la estrategia de chunking

        Args:
            notes_dir: Ruta al directorio de notas.
            show_progress: Mostrar barra de progreso (False en tests).

        Returns:
            IndexReport con desglose (procesadas, skipped, eliminadas).
        """
        start = time.time()
        report = IndexReport()

        # 1. Obtener qué está indexado actualmente.
        indexed = self.store.get_indexed_sources()

        # 2. Escanear notas en disco.
        paths = scan_notes(notes_dir)
        current_files: dict[str, Path] = {p.name: p for p in paths}

        # 3. Clasificar: nuevo, modificado, sin cambios.
        to_process: list[Path] = []

        for name, path in current_files.items():
            current_hash = compute_content_hash(path)

            if name not in indexed:
                # NUEVO: no está en el índice.
                to_process.append(path)
            elif indexed[name] != current_hash:
                # MODIFICADO: hash diferente → borrar chunks viejos y reprocesar.
                self.store.delete_by_source(name)
                to_process.append(path)
            else:
                # SIN CAMBIOS: skip.
                report.notes_skipped += 1

        # 4. Detectar ELIMINADOS: están en el índice pero no en disco.
        for source in indexed:
            if source not in current_files:
                self.store.delete_by_source(source)
                report.notes_deleted += 1

        # 5. Procesar nuevos y modificados.
        if to_process:
            iterator = _progress_iterator(
                to_process, "Procesando cambios...", show_progress
            )
            for path in iterator:
                try:
                    self._process_note(path, report)
                except Exception as e:
                    error_msg = f"{path.name}: {e}"
                    report.errors.append(error_msg)
                    logger.warning("Error indexando %s: %s", path.name, e)

        report.duration_seconds = time.time() - start
        logger.info(
            "Indexación incremental: %d procesadas, %d skipped, %d eliminadas, "
            "%d errores en %.1fs",
            report.notes_processed, report.notes_skipped, report.notes_deleted,
            len(report.errors), report.duration_seconds,
        )
        return report

    # =========================================================================
    # PROCESO DE UNA NOTA — Compartido entre full e incremental
    # =========================================================================

    def _process_note(self, path: Path, report: IndexReport) -> None:
        """
        Procesa una sola nota: parsear → trocear → embeder → almacenar.

        Si la nota produce 0 chunks (body vacío), se incrementa
        notes_processed pero no chunks_created.
        """
        # 1. Parsear frontmatter + body.
        parsed = parse_note(path)

        # 2. Trocear el body.
        chunks = self.chunker.chunk(parsed.body, parsed.metadata)

        report.notes_processed += 1

        if not chunks:
            # Nota sin contenido útil (vacía o solo frontmatter).
            logger.debug("Nota sin chunks: %s", path.name)
            return

        # 3. Generar embeddings para todos los chunks de esta nota.
        #    Prepend del título para enriquecer el contexto semántico del embedding.
        #    El texto almacenado en ChromaDB sigue siendo el original (sin título).
        title = parsed.metadata.title
        texts_for_embedding = [f"{title}\n\n{c.text}" for c in chunks]
        embeddings = self.embedder.embed_documents(texts_for_embedding)

        # 4. Guardar el hash del contenido en los metadatos para incremental.
        parsed.metadata.content_hash = compute_content_hash(path)

        # 5. Almacenar en ChromaDB.
        self.store.upsert_chunks(chunks, embeddings, parsed.metadata)

        report.chunks_created += len(chunks)
        logger.debug(
            "Nota %s: %d chunks indexados", path.name, len(chunks)
        )


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def compute_content_hash(filepath: Path) -> str:
    """
    Calcula un hash MD5 del contenido del fichero.

    Se usa para detectar si una nota cambió desde la última indexación.
    MD5 es suficiente aquí — no necesitamos seguridad criptográfica,
    solo detectar cambios en el contenido. MD5 es rápido y produce
    colisiones tan raramente que es prácticamente imposible con
    ficheros de texto de <100KB.
    """
    content = filepath.read_bytes()
    return hashlib.md5(content).hexdigest()


def _progress_iterator(items: list, description: str, show: bool):
    """
    Wraps una lista con barra de progreso de rich (si show=True).

    En tests usamos show=False para no contaminar la salida.
    En producción (CLI), show=True para dar feedback al usuario.

    Nota: no usamos yield aquí porque una función con yield es siempre
    un generator y `return items` no funciona como se espera.
    Usamos un wrapper iterable en su lugar.
    """
    if not show:
        return items

    return _ProgressIterable(items, description)


class _ProgressIterable:
    """Wrapper que muestra barra de progreso de rich al iterar."""

    def __init__(self, items: list, description: str):
        self._items = items
        self._description = description

    def __iter__(self):
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task(self._description, total=len(self._items))
            for item in self._items:
                yield item
                progress.advance(task)
