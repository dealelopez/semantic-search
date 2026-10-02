# =============================================================================
# Tests — Funciones nuevas (rondas 1 y 2 de mejoras)
# =============================================================================
# Cubre: deduplicate_by_source, format_results_json, format_results_explain,
# model_check, close/context manager, min_score, keyword search,
# export/import, validate_config, setup_logging, upsert validation,
# two-phase indexing title prepend.
# =============================================================================

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import chromadb
import pytest

from src.chunker import HybridChunker
from src.config import validate_config
from src.embeddings import OllamaEmbedder, OllamaError
from src.indexer import Indexer
from src.models import Chunk, NoteMetadata, SearchResult
from src.search import SearchEngine, deduplicate_by_source
from src.store import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(source="test.md", score=0.9, tags=None):
    return SearchResult(
        source=source, heading="## Sec", text="Texto de prueba.",
        score=score, note_type=tags[0] if tags else "nota",
        tags=tags or ["nota"], created="2026-09-15",
    )


def _make_store():
    client = chromadb.EphemeralClient()
    name = f"test_{uuid.uuid4().hex[:8]}"
    return VectorStore(collection_name=name, client=client)


def _make_metadata(source="t.md", tags=None):
    tags = tags or ["nota"]
    return NoteMetadata(
        source_path=source, title="Test", tags=tags,
        note_type=tags[0], created="2026-01-01",
    )


def _make_chunks(source="t.md", n=3):
    return [
        Chunk(text=f"Chunk {i} text.", heading_path=f"## S{i}",
              chunk_index=i, token_count=10, source_path=source)
        for i in range(n)
    ]


def _make_embeddings(n=3, dims=768):
    return [[0.1 * (i + 1)] * dims for i in range(n)]


# ---------------------------------------------------------------------------
# deduplicate_by_source
# ---------------------------------------------------------------------------

class TestDeduplicateBySource:

    def test_keeps_highest_score_per_source(self):
        results = [
            _make_result("a.md", 0.8),
            _make_result("a.md", 0.9),
            _make_result("b.md", 0.7),
        ]
        deduped = deduplicate_by_source(results)
        assert len(deduped) == 2
        sources = {r.source: r.score for r in deduped}
        assert sources["a.md"] == 0.9
        assert sources["b.md"] == 0.7

    def test_empty_list(self):
        assert deduplicate_by_source([]) == []

    def test_single_result(self):
        r = [_make_result("x.md", 0.5)]
        assert deduplicate_by_source(r) == r

    def test_ordered_by_score_desc(self):
        results = [
            _make_result("a.md", 0.6),
            _make_result("b.md", 0.9),
            _make_result("c.md", 0.3),
        ]
        deduped = deduplicate_by_source(results)
        scores = [r.score for r in deduped]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# format_results_json
# ---------------------------------------------------------------------------

class TestFormatResultsJson:

    def test_output_is_valid_json(self, capsys):
        engine = SearchEngine(
            embedder=Mock(spec=OllamaEmbedder),
            store=Mock(spec=VectorStore),
        )
        engine.format_results_json([_make_result()], 0.12)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total"] == 1
        assert data["elapsed_seconds"] == 0.12
        assert len(data["results"]) == 1

    def test_empty_results_json(self, capsys):
        engine = SearchEngine(
            embedder=Mock(spec=OllamaEmbedder),
            store=Mock(spec=VectorStore),
        )
        engine.format_results_json([], 0.01)
        data = json.loads(capsys.readouterr().out)
        assert data["total"] == 0
        assert data["results"] == []

    def test_spanish_chars(self, capsys):
        engine = SearchEngine(
            embedder=Mock(spec=OllamaEmbedder),
            store=Mock(spec=VectorStore),
        )
        r = _make_result("año.md")
        r.text = "Búsqueda semántica con ñ y acentos"
        engine.format_results_json([r], 0.1)
        data = json.loads(capsys.readouterr().out)
        assert "ñ" in data["results"][0]["text"]


# ---------------------------------------------------------------------------
# format_results_explain
# ---------------------------------------------------------------------------

class TestFormatResultsExplain:

    def test_shows_explain_header(self, capsys):
        mock_store = Mock(spec=VectorStore)
        mock_store.collection_stats.return_value = {
            "total_chunks": 100, "total_sources": 10, "collection_name": "test",
        }
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.model = "nomic-embed-text"
        engine = SearchEngine(embedder=mock_embedder, store=mock_store)
        engine.format_results_explain("test", [_make_result()], 0.5)
        out = capsys.readouterr().out
        assert "explain" in out.lower()
        assert "100" in out
        assert "nomic-embed-text" in out

    def test_shows_distance(self, capsys):
        mock_store = Mock(spec=VectorStore)
        mock_store.collection_stats.return_value = {
            "total_chunks": 5, "total_sources": 1, "collection_name": "t",
        }
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.model = "test-model"
        engine = SearchEngine(embedder=mock_embedder, store=mock_store)
        engine.format_results_explain("q", [_make_result(score=0.85)], 0.1)
        out = capsys.readouterr().out
        assert "0.85" in out
        assert "0.15" in out  # distance = 1 - 0.85


# ---------------------------------------------------------------------------
# OllamaEmbedder — model_check, close, context manager
# ---------------------------------------------------------------------------

class TestModelCheck:

    def test_model_check_success(self):
        embedder = OllamaEmbedder(base_url="http://fake:11434")
        with patch.object(embedder, "embed_query", return_value=[0.1] * 768):
            assert embedder.model_check() is True

    def test_model_check_failure(self):
        embedder = OllamaEmbedder(base_url="http://fake:11434")
        with patch.object(embedder, "embed_query", side_effect=OllamaError("nope")):
            assert embedder.model_check() is False


class TestCloseAndContextManager:

    def test_close_calls_client_close(self):
        embedder = OllamaEmbedder(base_url="http://fake:11434")
        with patch.object(embedder._client, "close") as mock_close:
            embedder.close()
            mock_close.assert_called_once()

    def test_context_manager(self):
        with OllamaEmbedder(base_url="http://fake:11434") as embedder:
            assert embedder is not None
        # no exception = success


# ---------------------------------------------------------------------------
# min_score filtering
# ---------------------------------------------------------------------------

class TestMinScore:

    def test_filters_below_threshold(self):
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_store = Mock(spec=VectorStore)
        mock_store.search.return_value = [
            _make_result(score=0.9),
            _make_result("b.md", score=0.4),
            _make_result("c.md", score=0.2),
        ]
        engine = SearchEngine(embedder=mock_embedder, store=mock_store)
        results, _ = engine.search("test", min_score=0.5)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_no_filter_when_zero(self):
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_store = Mock(spec=VectorStore)
        mock_store.search.return_value = [
            _make_result(score=0.3),
            _make_result("b.md", score=0.1),
        ]
        engine = SearchEngine(embedder=mock_embedder, store=mock_store)
        results, _ = engine.search("test", min_score=0.0)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# keyword (hybrid search)
# ---------------------------------------------------------------------------

class TestKeywordSearch:

    def test_keyword_passed_to_store(self):
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_store = Mock(spec=VectorStore)
        mock_store.search.return_value = []
        engine = SearchEngine(embedder=mock_embedder, store=mock_store)

        engine.search("test", keyword="HNSW")

        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs["where_document"] == {"$contains": "HNSW"}

    def test_no_keyword_none_document_filter(self):
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_store = Mock(spec=VectorStore)
        mock_store.search.return_value = []
        engine = SearchEngine(embedder=mock_embedder, store=mock_store)

        engine.search("test", keyword=None)

        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs["where_document"] is None


# ---------------------------------------------------------------------------
# VectorStore — upsert validation
# ---------------------------------------------------------------------------

class TestUpsertValidation:

    def test_mismatch_raises(self):
        store = _make_store()
        chunks = _make_chunks(n=3)
        embeddings = _make_embeddings(n=2)  # mismatch!
        with pytest.raises(ValueError, match="Mismatch"):
            store.upsert_chunks(chunks, embeddings, _make_metadata())


# ---------------------------------------------------------------------------
# VectorStore — export / import
# ---------------------------------------------------------------------------

class TestExportImport:

    def test_export_creates_valid_json(self, tmp_path):
        store = _make_store()
        chunks = _make_chunks(n=3)
        embs = _make_embeddings(n=3)
        store.upsert_chunks(chunks, embs, _make_metadata())

        filepath = tmp_path / "backup.json"
        count = store.export_collection(filepath)
        assert count == 3
        assert filepath.exists()

        data = json.loads(filepath.read_text())
        assert data["total_items"] == 3
        assert len(data["items"]) == 3
        assert "exported_at" in data

    def test_import_restores_data(self, tmp_path):
        # Export
        store1 = _make_store()
        chunks = _make_chunks(n=4)
        embs = _make_embeddings(n=4)
        store1.upsert_chunks(chunks, embs, _make_metadata())

        filepath = tmp_path / "backup.json"
        store1.export_collection(filepath)

        # Import into a fresh store
        store2 = _make_store()
        assert store2.collection_stats()["total_chunks"] == 0
        count = store2.import_collection(filepath)
        assert count == 4
        assert store2.collection_stats()["total_chunks"] == 4

    def test_export_empty_collection(self, tmp_path):
        store = _make_store()
        filepath = tmp_path / "empty.json"
        count = store.export_collection(filepath)
        assert count == 0
        data = json.loads(filepath.read_text())
        assert data["total_items"] == 0

    def test_import_empty_file(self, tmp_path):
        store = _make_store()
        filepath = tmp_path / "empty.json"
        filepath.write_text('{"items": []}')
        count = store.import_collection(filepath)
        assert count == 0


# ---------------------------------------------------------------------------
# VectorStore — where_document (keyword search)
# ---------------------------------------------------------------------------

class TestStoreKeywordSearch:

    def test_where_document_filters(self):
        store = _make_store()
        c1 = [Chunk(text="Python es genial para HNSW", heading_path="",
                     chunk_index=0, token_count=10, source_path="a.md")]
        c2 = [Chunk(text="Java es otro lenguaje", heading_path="",
                     chunk_index=0, token_count=10, source_path="b.md")]
        e1 = [[0.1] * 768]
        e2 = [[0.2] * 768]
        store.upsert_chunks(c1, e1, _make_metadata("a.md"))
        store.upsert_chunks(c2, e2, _make_metadata("b.md"))

        results = store.search(
            query_embedding=[0.15] * 768, n_results=5,
            where_document={"$contains": "HNSW"},
        )
        assert len(results) == 1
        assert "HNSW" in results[0].text


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:

    def test_default_config_valid(self):
        issues = validate_config()
        # Default config should be valid (or only have non-critical issues)
        for issue in issues:
            # URL validation might trigger on default "http://localhost:11434"
            assert "OLLAMA_BASE_URL" not in issue or "http" in issue

    def test_detects_bad_url(self):
        with patch("src.config.OLLAMA_BASE_URL", "192.0.2.1:11434"):
            issues = validate_config()
            assert any("OLLAMA_BASE_URL" in i for i in issues)

    def test_detects_empty_model(self):
        with patch("src.config.EMBEDDING_MODEL", ""):
            issues = validate_config()
            assert any("EMBEDDING_MODEL" in i for i in issues)

    def test_detects_bad_chunk_params(self):
        with patch("src.config.MAX_CHUNK_TOKENS", 50), \
             patch("src.config.MIN_CHUNK_TOKENS", 100):
            issues = validate_config()
            assert any("MAX_CHUNK_TOKENS" in i for i in issues)

    def test_detects_bad_overlap(self):
        with patch("src.config.OVERLAP_TOKENS", 500), \
             patch("src.config.MAX_CHUNK_TOKENS", 300):
            issues = validate_config()
            assert any("OVERLAP_TOKENS" in i for i in issues)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:

    def test_setup_logging_default(self):
        # Should not raise
        from src.config import setup_logging
        setup_logging(verbose=False)

    def test_setup_logging_verbose(self):
        import logging
        from src.config import setup_logging
        setup_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG


# ---------------------------------------------------------------------------
# CLI parser — nuevos flags
# ---------------------------------------------------------------------------

class TestCliNewFlags:

    def test_parse_min_score(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "query", "--min-score", "0.5"])
        assert args.min_score == 0.5

    def test_parse_keyword(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "query", "--keyword", "HNSW"])
        assert args.keyword == "HNSW"

    def test_parse_keyword_short(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "query", "-k", "redis"])
        assert args.keyword == "redis"

    def test_parse_full(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "query", "--full"])
        assert args.full is True

    def test_parse_verbose(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["-v", "search", "query"])
        assert args.verbose is True

    def test_parse_watch(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["watch", "--delay", "5"])
        assert args.command == "watch"
        assert args.delay == 5.0

    def test_parse_export(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["export", "/tmp/backup.json"])
        assert args.command == "export"
        assert args.filepath == "/tmp/backup.json"

    def test_parse_import(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["import", "/tmp/backup.json"])
        assert args.command == "import"

    def test_parse_validate(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["validate"])
        assert args.command == "validate"

    def test_parse_json(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "q", "--json"])
        assert args.json is True

    def test_parse_unique(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "q", "--unique"])
        assert args.unique is True

    def test_parse_explain(self):
        from src.cli import build_parser
        p = build_parser()
        args = p.parse_args(["search", "q", "--explain"])
        assert args.explain is True


# ---------------------------------------------------------------------------
# Indexer — title prepend in embeddings
# ---------------------------------------------------------------------------

class TestTitlePrepend:

    def test_embed_receives_title_prefixed_text(self):
        mock_embedder = Mock(spec=OllamaEmbedder)
        mock_embedder.embed_documents.return_value = [[0.1] * 768, [0.1] * 768]
        mock_store = Mock(spec=VectorStore)
        mock_store.get_indexed_sources.return_value = {}
        mock_store.upsert_chunks.return_value = 0
        mock_store.reset_collection.return_value = None

        indexer = Indexer(
            embedder=mock_embedder,
            store=mock_store,
            chunker=HybridChunker(),
        )

        report = indexer.full_reindex(str(FIXTURES), show_progress=False)
        assert report.notes_processed > 0

        # Verify that embed_documents was called with title-prefixed texts.
        call_args = mock_embedder.embed_documents.call_args
        texts = call_args[0][0]  # first positional arg
        # At least one text should contain a title prefix
        # (fixture nota_con_frontmatter.md has title "Embeddings y búsqueda semántica")
        has_title = any("Embeddings y búsqueda semántica" in t for t in texts)
        assert has_title, f"Expected title prefix in texts: {texts[:2]}"


# ---------------------------------------------------------------------------
# format_results max_preview (--full)
# ---------------------------------------------------------------------------

class TestFormatResultsFull:

    def test_full_text_not_truncated(self, capsys):
        engine = SearchEngine(
            embedder=Mock(spec=OllamaEmbedder),
            store=Mock(spec=VectorStore),
        )
        long_text = "x" * 500
        r = SearchResult(
            source="t.md", heading="", text=long_text,
            score=0.9, note_type="nota", tags=["nota"],
        )
        engine.format_results("q", [r], 0.1, max_preview=100_000)
        out = capsys.readouterr().out
        assert "..." not in out  # not truncated

    def test_default_preview_truncates(self, capsys):
        engine = SearchEngine(
            embedder=Mock(spec=OllamaEmbedder),
            store=Mock(spec=VectorStore),
        )
        long_text = "x" * 500
        r = SearchResult(
            source="t.md", heading="", text=long_text,
            score=0.9, note_type="nota", tags=["nota"],
        )
        engine.format_results("q", [r], 0.1)
        out = capsys.readouterr().out
        assert "..." in out  # truncated
