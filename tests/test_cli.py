# =============================================================================
# Tests — CLI (argparse parsing)
# =============================================================================
# Estos tests solo verifican el parsing de argumentos y la estructura
# del parser. No necesitan ChromaDB ni Ollama porque no ejecutan main().
# =============================================================================

import pytest

from src.cli import build_parser


@pytest.fixture
def parser():
    return build_parser()


# ---------------------------------------------------------------------------
# search — one-shot
# ---------------------------------------------------------------------------

class TestSearchOneshot:

    def test_parse_query(self, parser):
        args = parser.parse_args(["search", "mi query"])
        assert args.command == "search"
        assert args.query == "mi query"
        assert args.interactive is False

    def test_parse_con_n(self, parser):
        args = parser.parse_args(["search", "query", "-n", "10"])
        assert args.n == 10

    def test_parse_con_filtros(self, parser):
        args = parser.parse_args([
            "search", "query",
            "--tags", "ia,ml",
            "--type", "nota",
            "-n", "10",
        ])
        assert args.tags == "ia,ml"
        assert args.note_type == "nota"
        assert args.n == 10
        assert args.query == "query"


# ---------------------------------------------------------------------------
# search — interactive
# ---------------------------------------------------------------------------

class TestSearchInteractive:

    def test_parse_interactive(self, parser):
        args = parser.parse_args(["search", "--interactive"])
        assert args.interactive is True
        assert args.query is None

    def test_parse_interactive_short(self, parser):
        args = parser.parse_args(["search", "-i"])
        assert args.interactive is True


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

class TestIndex:

    def test_parse_full(self, parser):
        args = parser.parse_args(["index", "--mode", "full"])
        assert args.command == "index"
        assert args.mode == "full"

    def test_parse_incremental(self, parser):
        args = parser.parse_args(["index", "--mode", "incremental"])
        assert args.mode == "incremental"

    def test_mode_requerido(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["index"])

    def test_mode_invalido(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["index", "--mode", "invalido"])


# ---------------------------------------------------------------------------
# health y status
# ---------------------------------------------------------------------------

class TestOtrosSubcomandos:

    def test_health(self, parser):
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_status(self, parser):
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_sin_subcomando(self, parser):
        args = parser.parse_args([])
        assert args.command is None
