# =============================================================================
# Tests — VectorStore con ChromaDB efímero (sin Docker)
# =============================================================================
#
# Usamos chromadb.EphemeralClient() que crea una instancia en memoria.
# Esto permite testear todas las operaciones CRUD sin Docker ni red.
# EphemeralClient está en el paquete chromadb completo (requirements-dev.txt),
# no en chromadb-client (que es solo el cliente HTTP de producción).
# =============================================================================

import uuid

import chromadb
import pytest

from src.models import Chunk, NoteMetadata
from src.store import VectorStore, _deterministic_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """VectorStore con ChromaDB en memoria y colección única por test."""
    client = chromadb.EphemeralClient()
    # Nombre único para aislar cada test (EphemeralClient es singleton).
    name = f"test_{uuid.uuid4().hex[:8]}"
    return VectorStore(collection_name=name, client=client)


def _make_metadata(source: str = "nota-test.md", tags: list[str] | None = None) -> NoteMetadata:
    # None → tags por defecto; [] explícito se respeta (nota sin frontmatter).
    if tags is None:
        tags = ["nota", "ia"]
    return NoteMetadata(
        source_path=source,
        title="Test Note",
        tags=tags,
        note_type=tags[0] if tags else "sin-tipo",
        created="2026-09-15",
    )


def _make_chunks(source: str = "nota-test.md", n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            text=f"Contenido del chunk {i} de la nota {source}.",
            heading_path=f"## Sección {i}",
            chunk_index=i,
            token_count=20,
            source_path=source,
        )
        for i in range(n)
    ]


def _make_embeddings(n: int = 3, dims: int = 768) -> list[list[float]]:
    """Genera embeddings dummy (diferentes entre sí para tests de búsqueda)."""
    return [[0.1 * (i + 1)] * dims for i in range(n)]


# ---------------------------------------------------------------------------
# Upsert y count
# ---------------------------------------------------------------------------

class TestUpsert:

    def test_upsert_y_count(self, store):
        chunks = _make_chunks(n=3)
        embeddings = _make_embeddings(n=3)
        meta = _make_metadata()

        count = store.upsert_chunks(chunks, embeddings, meta)
        assert count == 3

        stats = store.collection_stats()
        assert stats["total_chunks"] == 3

    def test_upsert_idempotente(self, store):
        """Insertar los mismos chunks 2 veces → count sigue siendo 3."""
        chunks = _make_chunks(n=3)
        embeddings = _make_embeddings(n=3)
        meta = _make_metadata()

        store.upsert_chunks(chunks, embeddings, meta)
        store.upsert_chunks(chunks, embeddings, meta)

        stats = store.collection_stats()
        assert stats["total_chunks"] == 3

    def test_upsert_vacio(self, store):
        count = store.upsert_chunks([], [], _make_metadata())
        assert count == 0

    def test_upsert_sin_tags(self, store):
        """Nota sin frontmatter (tags=[]) → ChromaDB 1.5+ rechaza listas
        vacías, el store debe omitir la clave y no fallar."""
        chunks = _make_chunks(n=2)
        embeddings = _make_embeddings(n=2)
        meta = _make_metadata(tags=[])

        count = store.upsert_chunks(chunks, embeddings, meta)
        assert count == 2

        # La nota queda buscable aunque no tenga tags.
        results = store.search(query_embedding=[0.1] * 768, n_results=5)
        assert len(results) > 0
        assert results[0].tags == []

    def test_upsert_sin_content_hash(self, store):
        """content_hash=None no debe enviarse como None a ChromaDB."""
        chunks = _make_chunks(n=1)
        embeddings = _make_embeddings(n=1)
        meta = _make_metadata()
        meta.content_hash = None

        count = store.upsert_chunks(chunks, embeddings, meta)
        assert count == 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:

    def test_search_devuelve_resultados(self, store):
        chunks = _make_chunks(n=3)
        embeddings = _make_embeddings(n=3)
        store.upsert_chunks(chunks, embeddings, _make_metadata())

        # Buscar con un vector parecido al primer chunk.
        results = store.search(query_embedding=[0.1] * 768, n_results=3)
        assert len(results) > 0

    def test_search_devuelve_ordenado_por_score(self, store):
        chunks = _make_chunks(n=3)
        embeddings = _make_embeddings(n=3)
        store.upsert_chunks(chunks, embeddings, _make_metadata())

        results = store.search(query_embedding=[0.1] * 768, n_results=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_con_filtro_tipo(self, store):
        """Insertar notas y proyectos, filtrar por tipo."""
        # Notas
        chunks_nota = _make_chunks("nota.md", n=2)
        emb_nota = _make_embeddings(n=2)
        meta_nota = _make_metadata("nota.md", tags=["nota"])

        # Proyectos
        chunks_proy = _make_chunks("proy.md", n=2)
        emb_proy = _make_embeddings(n=2)
        meta_proy = _make_metadata("proy.md", tags=["proyecto"])

        store.upsert_chunks(chunks_nota, emb_nota, meta_nota)
        store.upsert_chunks(chunks_proy, emb_proy, meta_proy)

        results = store.search(
            query_embedding=[0.1] * 768,
            n_results=10,
            where_filter={"note_type": {"$eq": "nota"}},
        )
        for r in results:
            assert r.note_type == "nota"

    def test_search_con_filtro_tags(self, store):
        """Filtrar por tag usando $contains sobre array nativo."""
        chunks = _make_chunks(n=2)
        embeddings = _make_embeddings(n=2)
        meta = _make_metadata(tags=["nota", "ia", "embeddings"])
        store.upsert_chunks(chunks, embeddings, meta)

        results = store.search(
            query_embedding=[0.1] * 768,
            n_results=5,
            where_filter={"tags": {"$contains": "ia"}},
        )
        assert len(results) > 0
        for r in results:
            assert "ia" in r.tags

    def test_search_coleccion_vacia(self, store):
        results = store.search(query_embedding=[0.1] * 768, n_results=5)
        assert results == []

    def test_search_score_entre_0_y_1(self, store):
        chunks = _make_chunks(n=2)
        embeddings = _make_embeddings(n=2)
        store.upsert_chunks(chunks, embeddings, _make_metadata())

        results = store.search(query_embedding=[0.1] * 768, n_results=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:

    def test_delete_by_source(self, store):
        """Insertar chunks de 2 ficheros, borrar 1 → solo quedan los del otro."""
        chunks_a = _make_chunks("a.md", n=2)
        chunks_b = _make_chunks("b.md", n=3)
        emb_a = _make_embeddings(n=2)
        emb_b = _make_embeddings(n=3)

        store.upsert_chunks(chunks_a, emb_a, _make_metadata("a.md"))
        store.upsert_chunks(chunks_b, emb_b, _make_metadata("b.md"))

        assert store.collection_stats()["total_chunks"] == 5

        deleted = store.delete_by_source("a.md")
        assert deleted == 2
        assert store.collection_stats()["total_chunks"] == 3

    def test_delete_source_inexistente(self, store):
        deleted = store.delete_by_source("no-existe.md")
        assert deleted == 0


# ---------------------------------------------------------------------------
# Get indexed sources
# ---------------------------------------------------------------------------

class TestGetIndexedSources:

    def test_devuelve_sources_indexados(self, store):
        for name in ["a.md", "b.md", "c.md"]:
            chunks = _make_chunks(name, n=2)
            emb = _make_embeddings(n=2)
            store.upsert_chunks(chunks, emb, _make_metadata(name))

        sources = store.get_indexed_sources()
        assert len(sources) == 3
        assert "a.md" in sources
        assert "b.md" in sources
        assert "c.md" in sources

    def test_coleccion_vacia(self, store):
        assert store.get_indexed_sources() == {}


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_collection(self, store):
        chunks = _make_chunks(n=5)
        emb = _make_embeddings(n=5)
        store.upsert_chunks(chunks, emb, _make_metadata())

        assert store.collection_stats()["total_chunks"] == 5

        store.reset_collection()
        assert store.collection_stats()["total_chunks"] == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:

    def test_stats_con_datos(self, store):
        for name in ["x.md", "y.md"]:
            chunks = _make_chunks(name, n=3)
            emb = _make_embeddings(n=3)
            store.upsert_chunks(chunks, emb, _make_metadata(name))

        stats = store.collection_stats()
        assert stats["total_chunks"] == 6
        assert stats["total_sources"] == 2
        assert stats["collection_name"].startswith("test_")

    def test_stats_vacia(self, store):
        stats = store.collection_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_sources"] == 0


# ---------------------------------------------------------------------------
# _sanitize_metadata (función pura, sin ChromaDB)
# ---------------------------------------------------------------------------

class TestSanitizeMetadata:

    def test_elimina_lista_vacia_y_none(self):
        from src.store import _sanitize_metadata
        out = _sanitize_metadata({
            "source": "a.md",
            "tags": [],
            "content_hash": None,
            "note_type": "sin-tipo",
        })
        assert out == {"source": "a.md", "note_type": "sin-tipo"}

    def test_conserva_valores_validos(self):
        from src.store import _sanitize_metadata
        meta = {
            "source": "a.md",
            "tags": ["nota", "ia"],
            "note_type": "nota",
            "created": "2026-09-15",
            "content_hash": "abc123",
        }
        assert _sanitize_metadata(meta) == meta


# ---------------------------------------------------------------------------
# IDs deterministas
# ---------------------------------------------------------------------------

class TestDeterministicId:

    def test_mismo_input_mismo_id(self):
        id1 = _deterministic_id("nota.md", 0)
        id2 = _deterministic_id("nota.md", 0)
        assert id1 == id2

    def test_diferente_input_diferente_id(self):
        id1 = _deterministic_id("nota.md", 0)
        id2 = _deterministic_id("nota.md", 1)
        assert id1 != id2

    def test_id_es_hex_string(self):
        result = _deterministic_id("test.md", 5)
        assert len(result) == 32  # MD5 hex
        assert all(c in "0123456789abcdef" for c in result)
