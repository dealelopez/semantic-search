# =============================================================================
# Tests — Frontmatter parser y escaneo de notas
# =============================================================================

from pathlib import Path

from src.frontmatter_parser import parse_note, scan_notes

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# parse_note — con frontmatter
# ---------------------------------------------------------------------------

class TestParseConFrontmatter:

    def test_titulo_del_yaml(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.title == "Embeddings y búsqueda semántica"

    def test_tags_completos(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.tags == [
            "nota",
            "inteligencia-artificial",
            "embeddings",
            "estado/seedling",
        ]

    def test_note_type_es_primer_tag(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.note_type == "nota"

    def test_fecha_created(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.created == "2026-09-15"

    def test_fecha_updated(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.updated == "2026-09-20"

    def test_extra_fields_incluye_status(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.extra_fields.get("status") == "activo"

    def test_source_path_es_nombre_fichero(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert result.metadata.source_path == "nota_con_frontmatter.md"


# ---------------------------------------------------------------------------
# parse_note — sin frontmatter
# ---------------------------------------------------------------------------

class TestParseSinFrontmatter:

    def test_titulo_es_nombre_fichero(self):
        result = parse_note(FIXTURES / "nota_sin_frontmatter.md")
        assert result.metadata.title == "nota_sin_frontmatter"

    def test_tags_vacios(self):
        result = parse_note(FIXTURES / "nota_sin_frontmatter.md")
        assert result.metadata.tags == []

    def test_note_type_sin_tipo(self):
        result = parse_note(FIXTURES / "nota_sin_frontmatter.md")
        assert result.metadata.note_type == "sin-tipo"

    def test_body_es_todo_el_contenido(self):
        result = parse_note(FIXTURES / "nota_sin_frontmatter.md")
        assert result.body.startswith("# Nota suelta sin metadatos")


# ---------------------------------------------------------------------------
# Limpieza de wikilinks
# ---------------------------------------------------------------------------

class TestWikilinks:

    def test_wikilinks_eliminados_del_body(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert "[[RAG" not in result.body
        assert "RAG — Retrieval-Augmented Generation" in result.body

    def test_wikilink_con_alias(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert "[[nomic-embed-text|nomic]]" not in result.body
        assert "nomic" in result.body

    def test_wikilink_simple_en_nota_sin_frontmatter(self):
        result = parse_note(FIXTURES / "nota_sin_frontmatter.md")
        assert "[[enlace wiki]]" not in result.body
        assert "enlace wiki" in result.body


# ---------------------------------------------------------------------------
# Body limpio (sin YAML)
# ---------------------------------------------------------------------------

class TestBodyLimpio:

    def test_body_no_contiene_delimitador_yaml(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert not result.body.strip().startswith("---")

    def test_body_no_contiene_campos_yaml(self):
        result = parse_note(FIXTURES / "nota_con_frontmatter.md")
        assert "tags:" not in result.body
        assert "created:" not in result.body


# ---------------------------------------------------------------------------
# scan_notes — escaneo de directorio
# ---------------------------------------------------------------------------

class TestScanNotes:

    def test_encuentra_ficheros_md(self):
        paths = scan_notes(FIXTURES)
        nombres = [p.name for p in paths]
        assert "nota_con_frontmatter.md" in nombres
        assert "nota_sin_frontmatter.md" in nombres

    def test_devuelve_paths_ordenados(self):
        paths = scan_notes(FIXTURES)
        assert paths == sorted(paths)

    def test_directorio_inexistente_devuelve_vacio(self):
        paths = scan_notes(Path("/ruta/que/no/existe"))
        assert paths == []
