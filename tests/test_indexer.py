# =============================================================================
# Tests — Indexer (orquestación de indexación)
# =============================================================================
# Todos los tests usan mocks de embedder y store para no depender de
# Docker ni Ollama. El chunker es real (HybridChunker) porque es rápido
# y no tiene dependencias externas.
# =============================================================================

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.chunker import HybridChunker
from src.embeddings import OllamaEmbedder
from src.indexer import Indexer, compute_content_hash
from src.store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    """Embedder que devuelve vectores dummy de 768 dims."""
    embedder = Mock(spec=OllamaEmbedder)
    # embed_documents devuelve un vector por texto.
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    return embedder


@pytest.fixture
def mock_store():
    """Store con comportamiento básico mockeado."""
    store = Mock(spec=VectorStore)
    store.get_indexed_sources.return_value = {}
    store.upsert_chunks.return_value = 0
    store.delete_by_source.return_value = 0
    return store


@pytest.fixture
def chunker():
    return HybridChunker()


@pytest.fixture
def indexer(mock_embedder, mock_store, chunker):
    return Indexer(embedder=mock_embedder, store=mock_store, chunker=chunker)


# ---------------------------------------------------------------------------
# Full reindex
# ---------------------------------------------------------------------------

class TestFullReindex:

    def test_procesa_todas_las_notas(self, indexer):
        """Directorio con fixtures → procesa todas las notas .md."""
        report = indexer.full_reindex(str(FIXTURES), show_progress=False)
        # fixtures tiene 4 ficheros .md
        assert report.notes_processed >= 3

    def test_resetea_coleccion(self, indexer, mock_store):
        """Debe llamar a reset_collection antes de procesar."""
        indexer.full_reindex(str(FIXTURES), show_progress=False)
        mock_store.reset_collection.assert_called_once()

    def test_genera_chunks(self, indexer):
        report = indexer.full_reindex(str(FIXTURES), show_progress=False)
        assert report.chunks_created > 0

    def test_llama_embedder(self, indexer, mock_embedder):
        indexer.full_reindex(str(FIXTURES), show_progress=False)
        assert mock_embedder.embed_documents.call_count >= 1

    def test_llama_upsert(self, indexer, mock_store):
        indexer.full_reindex(str(FIXTURES), show_progress=False)
        assert mock_store.upsert_chunks.call_count >= 1

    def test_duration_positiva(self, indexer):
        report = indexer.full_reindex(str(FIXTURES), show_progress=False)
        assert report.duration_seconds > 0

    def test_directorio_vacio(self, indexer, tmp_path):
        report = indexer.full_reindex(str(tmp_path), show_progress=False)
        assert report.notes_processed == 0
        assert report.chunks_created == 0


# ---------------------------------------------------------------------------
# Incremental index
# ---------------------------------------------------------------------------

class TestIncrementalIndex:

    def test_detecta_nuevos(self, indexer, mock_store):
        """Store vacío + notas en disco → todas son nuevas."""
        mock_store.get_indexed_sources.return_value = {}

        report = indexer.incremental_index(str(FIXTURES), show_progress=False)
        assert report.notes_processed >= 3
        assert report.notes_skipped == 0

    def test_skip_sin_cambios(self, indexer, mock_store):
        """Store con mismos hashes que disco → todas son skipped."""
        # Construir dict con hashes reales de los fixtures.
        sources = {}
        for p in sorted(FIXTURES.glob("*.md")):
            sources[p.name] = compute_content_hash(p)
        mock_store.get_indexed_sources.return_value = sources

        report = indexer.incremental_index(str(FIXTURES), show_progress=False)
        assert report.notes_skipped == len(sources)
        assert report.notes_processed == 0

    def test_detecta_modificados(self, indexer, mock_store, tmp_path):
        """Store con hash viejo, disco con hash nuevo → reprocessa."""
        # Crear un fichero temporal.
        nota = tmp_path / "test.md"
        nota.write_text("---\ntitle: Test\ntags:\n  - nota\n---\n\n# Hola\n\nContenido original.")

        # Store dice que tiene este fichero con un hash diferente.
        mock_store.get_indexed_sources.return_value = {"test.md": "hash_viejo"}

        report = indexer.incremental_index(str(tmp_path), show_progress=False)
        assert report.notes_processed == 1
        # Debe borrar los chunks viejos antes de insertar los nuevos.
        mock_store.delete_by_source.assert_called_with("test.md")

    def test_detecta_eliminados(self, indexer, mock_store, tmp_path):
        """Store tiene 'borrada.md' pero no está en disco → eliminada."""
        mock_store.get_indexed_sources.return_value = {"borrada.md": "hash123"}

        report = indexer.incremental_index(str(tmp_path), show_progress=False)
        assert report.notes_deleted == 1
        mock_store.delete_by_source.assert_called_with("borrada.md")


# ---------------------------------------------------------------------------
# Resiliencia a errores
# ---------------------------------------------------------------------------

class TestErrores:

    def test_error_en_nota_no_para_proceso(self, mock_store, chunker):
        """Si el embedder falla para una nota, las demás se procesan."""
        call_count = 0

        def flaky_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Ollama timeout simulado")
            return [[0.1] * 768 for _ in texts]

        embedder = Mock(spec=OllamaEmbedder)
        embedder.embed_documents.side_effect = flaky_embed

        mock_store.get_indexed_sources.return_value = {}
        indexer = Indexer(embedder=embedder, store=mock_store, chunker=chunker)

        report = indexer.full_reindex(str(FIXTURES), show_progress=False)
        # Al menos una nota debe haberse procesado, y al menos un error.
        assert report.notes_processed >= 1
        assert len(report.errors) >= 1

    def test_error_en_upsert_no_aborta_full(self, mock_embedder, mock_store, chunker):
        """Si el upsert de una nota falla (ej: metadata rechazada por
        ChromaDB), las demás notas se guardan igual y el error queda
        registrado en vez de propagarse."""
        calls = {"n": 0}

        def flaky_upsert(chunks, embeddings, metadata):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError(
                    "Expected metadata list value for key 'tags' to be non-empty"
                )
            return len(chunks)

        mock_store.upsert_chunks.side_effect = flaky_upsert
        indexer = Indexer(embedder=mock_embedder, store=mock_store, chunker=chunker)

        report = indexer.full_reindex(str(FIXTURES), show_progress=False)
        assert len(report.errors) == 1
        assert "tags" in report.errors[0]
        assert report.chunks_created > 0


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------

class TestContentHash:

    def test_hash_determinista(self):
        path = FIXTURES / "nota_con_frontmatter.md"
        h1 = compute_content_hash(path)
        h2 = compute_content_hash(path)
        assert h1 == h2

    def test_hash_es_hex_md5(self):
        path = FIXTURES / "nota_con_frontmatter.md"
        h = compute_content_hash(path)
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_archivos_diferentes_hash_diferente(self):
        h1 = compute_content_hash(FIXTURES / "nota_con_frontmatter.md")
        h2 = compute_content_hash(FIXTURES / "nota_sin_frontmatter.md")
        assert h1 != h2
