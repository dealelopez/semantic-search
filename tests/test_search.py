# =============================================================================
# Tests — Motor de búsqueda semántica
# =============================================================================

from unittest.mock import Mock

import pytest

from src.embeddings import OllamaEmbedder
from src.models import SearchResult
from src.search import SearchEngine, _build_where_filter
from src.store import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    embedder = Mock(spec=OllamaEmbedder)
    embedder.embed_query.return_value = [0.1] * 768
    return embedder


@pytest.fixture
def mock_store():
    store = Mock(spec=VectorStore)
    store.search.return_value = []
    return store


@pytest.fixture
def engine(mock_embedder, mock_store):
    return SearchEngine(embedder=mock_embedder, store=mock_store)


def _make_result(source="test.md", score=0.9, tags=None):
    return SearchResult(
        source=source,
        heading="## Sección",
        text="Contenido de prueba con texto suficiente para un preview.",
        score=score,
        note_type=tags[0] if tags else "nota",
        tags=tags or ["nota"],
        created="2026-09-15",
    )


# ---------------------------------------------------------------------------
# search — sin filtros
# ---------------------------------------------------------------------------

class TestSearchSinFiltros:

    def test_llama_embed_query(self, engine, mock_embedder):
        engine.search("mi consulta")
        mock_embedder.embed_query.assert_called_once_with("mi consulta")

    def test_llama_store_search(self, engine, mock_store):
        engine.search("mi consulta")
        mock_store.search.assert_called_once()

    def test_sin_filtros_where_none(self, engine, mock_store):
        engine.search("consulta")
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs.get("where_filter") is None

    def test_devuelve_resultados_y_elapsed(self, engine, mock_store):
        mock_store.search.return_value = [_make_result()]
        results, elapsed = engine.search("test")
        assert len(results) == 1
        assert elapsed > 0


# ---------------------------------------------------------------------------
# search — con filtros
# ---------------------------------------------------------------------------

class TestSearchConFiltros:

    def test_filtro_tipo(self, engine, mock_store):
        engine.search("consulta", filter_type="proyecto")
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs["where_filter"] == {"note_type": {"$eq": "proyecto"}}

    def test_filtro_un_tag(self, engine, mock_store):
        engine.search("consulta", filter_tags=["ia"])
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs["where_filter"] == {"tags": {"$contains": "ia"}}

    def test_filtro_multiples_tags(self, engine, mock_store):
        engine.search("consulta", filter_tags=["ia", "ml"])
        call_kwargs = mock_store.search.call_args.kwargs
        where = call_kwargs["where_filter"]
        assert "$and" in where
        assert {"tags": {"$contains": "ia"}} in where["$and"]
        assert {"tags": {"$contains": "ml"}} in where["$and"]

    def test_filtro_tipo_y_tags(self, engine, mock_store):
        engine.search("consulta", filter_type="nota", filter_tags=["ia"])
        call_kwargs = mock_store.search.call_args.kwargs
        where = call_kwargs["where_filter"]
        assert "$and" in where
        assert {"note_type": {"$eq": "nota"}} in where["$and"]
        assert {"tags": {"$contains": "ia"}} in where["$and"]


# ---------------------------------------------------------------------------
# format_results — output
# ---------------------------------------------------------------------------

class TestFormatResults:

    def test_resultados_vacios(self, engine, capsys):
        engine.format_results("test", [], 0.1)
        captured = capsys.readouterr()
        assert "No se encontraron resultados" in captured.out

    def test_trunca_preview(self, engine, capsys):
        largo = SearchResult(
            source="test.md", heading="", text="x" * 500,
            score=0.9, note_type="nota", tags=["nota"],
        )
        engine.format_results("test", [largo], 0.1)
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_muestra_source(self, engine, capsys):
        engine.format_results("test", [_make_result(source="mi-nota.md")], 0.1)
        captured = capsys.readouterr()
        assert "mi-nota.md" in captured.out

    def test_muestra_score(self, engine, capsys):
        engine.format_results("test", [_make_result(score=0.92)], 0.1)
        captured = capsys.readouterr()
        assert "0.92" in captured.out

    def test_muestra_tags(self, engine, capsys):
        engine.format_results("test", [_make_result(tags=["nota", "ia"])], 0.1)
        captured = capsys.readouterr()
        assert "nota" in captured.out
        assert "ia" in captured.out

    def test_muestra_total_y_tiempo(self, engine, capsys):
        results = [_make_result(), _make_result(source="otra.md")]
        engine.format_results("test", results, 0.34)
        captured = capsys.readouterr()
        assert "2 resultado(s)" in captured.out
        assert "0.34s" in captured.out

    def test_caracteres_especiales(self, engine, capsys):
        """Texto con emojis, acentos, ñ → no explota."""
        result = SearchResult(
            source="año-ñ-café.md", heading="## Café ☕",
            text="La búsqueda semántica es útil 🎉",
            score=0.85, note_type="nota", tags=["café"],
        )
        engine.format_results("búsqueda", [result], 0.1)
        captured = capsys.readouterr()
        assert "café" in captured.out


# ---------------------------------------------------------------------------
# _build_where_filter — unit tests
# ---------------------------------------------------------------------------

class TestBuildWhereFilter:

    def test_sin_filtros(self):
        assert _build_where_filter(None, None) is None

    def test_solo_tipo(self):
        result = _build_where_filter("nota", None)
        assert result == {"note_type": {"$eq": "nota"}}

    def test_solo_un_tag(self):
        result = _build_where_filter(None, ["ia"])
        assert result == {"tags": {"$contains": "ia"}}

    def test_multiples_tags(self):
        result = _build_where_filter(None, ["ia", "ml"])
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_tipo_y_tags(self):
        result = _build_where_filter("proyecto", ["ia"])
        assert "$and" in result
        assert len(result["$and"]) == 2
