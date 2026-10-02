# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# store.py — Capa de persistencia con ChromaDB (base de datos vectorial)
# =============================================================================
#
# Qué es una colección vectorial:
# ---
# Es el equivalente a una "tabla" en bases de datos relacionales, pero
# optimizada para buscar por similitud entre vectores. Cada entrada tiene:
#   - ID: identificador único del chunk
#   - Vector (embedding): lista de 768 floats que representa el significado
#   - Document: el texto original del chunk (para devolverlo en búsquedas)
#   - Metadata: key-value pairs filtrables (tags, tipo, fecha, fuente...)
#
# HNSW (Hierarchical Navigable Small World):
# ---
# El algoritmo que ChromaDB usa internamente para buscar vecinos cercanos.
# Sin HNSW, buscar el vector más parecido entre 10.000 requeriría comparar
# con todos (O(n)). HNSW lo hace en O(log n) construyendo un grafo
# navegable por capas — como un "mapa" del espacio vectorial con atajos.
# Nosotros no tocamos HNSW directamente, solo le indicamos que use
# distancia coseno al crear la colección.
#
# Espacio coseno:
# ---
# Al crear la colección con hnsw:space=cosine, le decimos a ChromaDB que
# compare vectores usando distancia coseno. Esta mide el ángulo entre
# dos vectores, no su magnitud. Esto es ideal para embeddings de texto
# porque textos de diferente longitud pueden tener el mismo significado
# (y por tanto la misma dirección en el espacio vectorial).
# Alternativas: l2 (euclídea) e ip (producto punto).
#
# IDs deterministas:
# ---
# Generamos IDs como md5(source_path + ":" + chunk_index). Esto permite
# hacer upsert idempotente: si reindexas la misma nota, los chunks
# existentes se sobreescriben en lugar de duplicarse. Es la base de la
# indexación incremental y la idempotencia del sistema.
# =============================================================================

import hashlib
import logging
from pathlib import Path

import chromadb

from src.models import Chunk, NoteMetadata, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Encapsula todas las operaciones con ChromaDB.

    Métodos principales:
    - upsert_chunks: insertar/actualizar chunks con embeddings y metadata
    - search: buscar por similitud vectorial con filtros opcionales
    - get_indexed_sources: listar qué notas están indexadas (para incremental)
    - delete_by_source: borrar todos los chunks de una nota
    - collection_stats: estadísticas de la colección
    - reset_collection: borrar y recrear la colección (para full reindex)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        client: chromadb.ClientAPI | None = None,
    ):
        """
        Conecta con ChromaDB y obtiene (o crea) la colección.

        Args:
            host: Hostname de ChromaDB ('chromadb' en Docker, 'localhost' fuera).
            port: Puerto de ChromaDB (default 8000).
            collection_name: Nombre de la colección (default 'notas').
            client: Cliente ChromaDB inyectado (para tests con EphemeralClient).
        """
        from src import config

        self._collection_name = collection_name or config.COLLECTION_NAME

        if client is not None:
            # Cliente inyectado (tests con EphemeralClient).
            self._client = client
        else:
            # Cliente HTTP para servidor real.
            _host = host or config.CHROMA_HOST
            _port = port or config.CHROMA_PORT
            self._client = chromadb.HttpClient(host=_host, port=_port)

        # get_or_create: si la colección ya existe, la obtiene;
        # si no, la crea. Idempotente y seguro.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # =========================================================================
    # UPSERT — Insertar o actualizar chunks
    # =========================================================================

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        note_metadata: NoteMetadata,
    ) -> int:
        """
        Inserta o actualiza chunks en ChromaDB.

        Usa IDs deterministas (hash de source_path + chunk_index) para que
        un upsert de la misma nota sobreescriba los chunks existentes sin
        crear duplicados. Esto es clave para la idempotencia.

        Args:
            chunks: Lista de chunks con texto, heading, etc.
            embeddings: Vectores correspondientes (mismo orden que chunks).
            note_metadata: Metadatos de la nota (tags, tipo, fecha...).

        Returns:
            Número de chunks insertados/actualizados.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            # ID determinista: hash del path + posición.
            # Permite upsert idempotente — misma nota → mismos IDs.
            chunk_id = _deterministic_id(chunk.source_path, chunk.chunk_index)
            ids.append(chunk_id)
            documents.append(chunk.text)

            # Metadata filtrable en búsquedas.
            meta: dict = {
                "source": chunk.source_path,
                "heading": chunk.heading_path,
                "note_type": note_metadata.note_type,
                "tags": note_metadata.tags,
            }
            if note_metadata.created:
                meta["created"] = note_metadata.created
            if hasattr(note_metadata, "content_hash"):
                meta["content_hash"] = note_metadata.content_hash

            metadatas.append(meta)

        # ChromaDB upsert: si el ID existe, sobreescribe; si no, inserta.
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(
            "Upsert %d chunks de %s", len(chunks), chunks[0].source_path
        )
        return len(chunks)

    # =========================================================================
    # SEARCH — Búsqueda por similitud vectorial
    # =========================================================================

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where_filter: dict | None = None,
        where_document: dict | None = None,
    ) -> list[SearchResult]:
        """
        Busca los chunks más similares al vector de la query.

        ChromaDB compara el query_embedding con todos los vectores de la
        colección (usando HNSW para hacerlo eficiente) y devuelve los N
        más cercanos según distancia coseno.

        Args:
            query_embedding: Vector de la query (768 floats).
            n_results: Número de resultados a devolver.
            where_filter: Filtro de metadata (ej: {"note_type": {"$eq": "nota"}}).
            where_document: Filtro sobre texto del documento
                            (ej: {"$contains": "HNSW"} para búsqueda híbrida).

        Returns:
            Lista de SearchResult ordenada por score descendente.
        """
        # Verificar que hay datos antes de buscar.
        if self._collection.count() == 0:
            return []

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count()),
        }
        if where_filter:
            query_kwargs["where"] = where_filter
        if where_document:
            query_kwargs["where_document"] = where_document

        results = self._collection.query(**query_kwargs)

        # ChromaDB devuelve listas anidadas (para soportar queries batch).
        # Como enviamos una sola query, accedemos al primer elemento.
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        search_results: list[SearchResult] = []

        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            text = documents[i] if i < len(documents) else ""

            # Convertir distancia a score de similitud.
            # Distancia coseno: 0 = idéntico, 2 = opuesto.
            # Score: 1 - distance. Para resultados relevantes, score > 0.5.
            score = max(0.0, 1.0 - distance)

            # Tags: ChromaDB 1.5+ devuelve arrays nativos.
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            search_results.append(SearchResult(
                source=meta.get("source", ""),
                heading=meta.get("heading", ""),
                text=text,
                score=round(score, 4),
                note_type=meta.get("note_type", "sin-tipo"),
                tags=tags,
                created=meta.get("created"),
            ))

        # Ordenar por score descendente (mayor similitud primero).
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    # =========================================================================
    # INDEXED SOURCES — Para indexación incremental
    # =========================================================================

    def get_indexed_sources(self) -> dict[str, str]:
        """
        Devuelve las notas indexadas y sus hashes de contenido.

        Se usa en indexación incremental para saber qué ya está indexado
        y si cambió (comparando hashes).

        Returns:
            Dict {source_path: content_hash}. El hash puede ser "" si
            la nota fue indexada sin hash (ej: versión anterior del indexer).
        """
        total = self._collection.count()
        if total == 0:
            return {}

        # Paginar para no cargar todo en memoria de golpe en colecciones grandes.
        sources: dict[str, str] = {}
        batch_size = 5000
        offset = 0

        while offset < total:
            data = self._collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            metadatas = data.get("metadatas", [])
            if not metadatas:
                break

            for meta in metadatas:
                source = meta.get("source", "")
                content_hash = meta.get("content_hash", "")
                if source and source not in sources:
                    sources[source] = content_hash

            offset += batch_size

        return sources

    # =========================================================================
    # DELETE — Borrar chunks de una nota
    # =========================================================================

    def delete_by_source(self, source_path: str) -> int:
        """
        Borra todos los chunks de una nota específica.

        Se usa cuando:
        - Un fichero cambió (se borran chunks viejos antes de insertar nuevos)
        - Un fichero fue eliminado del disco

        Args:
            source_path: Ruta de la nota (ej: "proyecto-x.md").

        Returns:
            Número de chunks eliminados.
        """
        # Buscar IDs con esa fuente.
        results = self._collection.get(
            where={"source": {"$eq": source_path}},
            include=[],
        )
        ids = results.get("ids", [])

        if not ids:
            return 0

        self._collection.delete(ids=ids)
        logger.debug("Eliminados %d chunks de %s", len(ids), source_path)
        return len(ids)

    # =========================================================================
    # STATS — Estadísticas de la colección
    # =========================================================================

    def collection_stats(self) -> dict:
        """
        Devuelve estadísticas de la colección.

        Returns:
            Dict con total_chunks, total_sources, y collection_name.
        """
        total_chunks = self._collection.count()

        # Contar fuentes únicas, paginando para colecciones grandes.
        sources: set[str] = set()
        if total_chunks > 0:
            batch_size = 5000
            offset = 0
            while offset < total_chunks:
                data = self._collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
                metadatas = data.get("metadatas", [])
                if not metadatas:
                    break
                sources.update(
                    m.get("source") for m in metadatas if m.get("source")
                )
                offset += batch_size

        return {
            "total_chunks": total_chunks,
            "total_sources": len(sources),
            "collection_name": self._collection_name,
        }

    # =========================================================================
    # EXPORT / IMPORT — Backup y restauración del índice
    # =========================================================================

    def export_collection(self, filepath: str | Path) -> int:
        """
        Exporta la colección completa a un fichero JSON.

        Incluye IDs, documentos, metadatos y embeddings. Permite restaurar
        el índice sin recomputar embeddings (que es la parte lenta).

        Args:
            filepath: Ruta del fichero JSON de destino.

        Returns:
            Número de chunks exportados.
        """
        import json
        from datetime import datetime, timezone

        filepath = Path(filepath)
        total = self._collection.count()
        if total == 0:
            # Exportar fichero vacío válido.
            output = {
                "collection_name": self._collection_name,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_items": 0,
                "items": [],
            }
            filepath.write_text(json.dumps(output, ensure_ascii=False, indent=2))
            return 0

        all_items: list[dict] = []
        batch_size = 1000
        offset = 0

        while offset < total:
            data = self._collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=batch_size,
                offset=offset,
            )
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            embeddings = data.get("embeddings", [])

            if not ids:
                break

            for i, item_id in enumerate(ids):
                all_items.append({
                    "id": item_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "embedding": (
                        embeddings[i].tolist()
                        if hasattr(embeddings[i], "tolist")
                        else list(embeddings[i])
                    ) if i < len(embeddings) else [],
                })

            offset += batch_size

        output = {
            "collection_name": self._collection_name,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_items": len(all_items),
            "items": all_items,
        }
        filepath.write_text(json.dumps(output, ensure_ascii=False))
        logger.info("Exportados %d chunks a %s", len(all_items), filepath)
        return len(all_items)

    def import_collection(self, filepath: str | Path, reset_first: bool = True) -> int:
        """
        Importa una colección desde un fichero JSON exportado previamente.

        Args:
            filepath: Ruta del fichero JSON de origen.
            reset_first: Si True, borra la colección antes de importar.

        Returns:
            Número de chunks importados.
        """
        import json

        filepath = Path(filepath)
        data = json.loads(filepath.read_text())
        items = data.get("items", [])

        if not items:
            return 0

        if reset_first:
            self.reset_collection()

        # Importar en lotes para no saturar ChromaDB.
        batch_size = 500
        total_imported = 0

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            self._collection.upsert(
                ids=[it["id"] for it in batch],
                documents=[it["document"] for it in batch],
                metadatas=[it["metadata"] for it in batch],
                embeddings=[it["embedding"] for it in batch],
            )
            total_imported += len(batch)

        logger.info("Importados %d chunks desde %s", total_imported, filepath)
        return total_imported

    # =========================================================================
    # RESET — Borrar y recrear la colección
    # =========================================================================

    def reset_collection(self) -> None:
        """
        Borra la colección entera y la recrea vacía.

        Se usa para reindexación completa (empezar de cero).
        Después de reset, collection_stats().total_chunks == 0.
        """
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Colección '%s' reseteada", self._collection_name)


# =============================================================================
# HELPER — IDs deterministas
# =============================================================================

def _deterministic_id(source_path: str, chunk_index: int) -> str:
    """
    Genera un ID determinista para un chunk.

    md5(source_path + ":" + chunk_index) produce el mismo ID cada vez
    para el mismo chunk. Esto permite:
    - Upsert idempotente (reindexar sin duplicar)
    - Identificar chunks específicos para borrado selectivo

    No usamos MD5 por seguridad criptográfica (para eso no sirve),
    sino como hash rápido y uniforme para generar IDs cortos.
    """
    raw = f"{source_path}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()
