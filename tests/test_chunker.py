# =============================================================================
# Tests — Chunker híbrido de Markdown
# =============================================================================

from pathlib import Path

from src.chunker import (
    HybridChunker,
    _split_by_headings,
    _split_by_paragraphs,
    _split_by_size,
    _group_small_paragraphs,
    estimate_tokens,
)
from src.frontmatter_parser import parse_note
from src.models import NoteMetadata

FIXTURES = Path(__file__).parent / "fixtures"


def _make_metadata(source: str = "test.md") -> NoteMetadata:
    """Helper para crear metadata mínima de test."""
    return NoteMetadata(source_path=source, title="Test")


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:

    def test_texto_vacio(self):
        assert estimate_tokens("") == 0

    def test_estimacion_razonable(self):
        # 10 palabras × 1.3 = 13 tokens
        texto = "una dos tres cuatro cinco seis siete ocho nueve diez"
        result = estimate_tokens(texto)
        assert result == 13

    def test_texto_largo_proporcional(self):
        texto = " ".join(["palabra"] * 100)
        result = estimate_tokens(texto)
        assert 120 <= result <= 140  # ~130 ± margen


# ---------------------------------------------------------------------------
# HybridChunker — por headings
# ---------------------------------------------------------------------------

class TestChunkPorHeadings:

    def test_secciones_generan_chunks(self):
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        assert len(chunks) > 0

    def test_heading_path_presente(self):
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        # Al menos un chunk debe tener heading_path no vacío.
        headings = [c.heading_path for c in chunks if c.heading_path]
        assert len(headings) > 0

    def test_heading_path_multinivel(self):
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        # La sección "### Router y DNS" está bajo "## Configuración de red"
        multi = [c for c in chunks if ">" in c.heading_path]
        assert len(multi) > 0, "Debe haber headings multinivel (## > ###)"

    def test_subdivision_seccion_larga(self):
        """Una sección >300 tokens debe subdividirse en chunks más pequeños."""
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker(max_tokens=300)
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        for c in chunks:
            # Permitir un margen del 30% porque estimate_tokens es aproximado.
            assert c.token_count <= 300 * 1.4, (
                f"Chunk demasiado grande: {c.token_count} tokens "
                f"(heading: {c.heading_path})"
            )


# ---------------------------------------------------------------------------
# HybridChunker — por párrafos (sin headings)
# ---------------------------------------------------------------------------

class TestChunkPorParrafos:

    def test_nota_sin_headings_genera_chunks(self):
        parsed = parse_note(FIXTURES / "nota_corta_sin_headings.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        assert len(chunks) > 0

    def test_agrupacion_parrafos_cortos(self):
        """Párrafos cortos consecutivos se agrupan hasta alcanzar min_tokens."""
        parsed = parse_note(FIXTURES / "nota_corta_sin_headings.md")
        # min_tokens alto para forzar agrupación.
        chunker = HybridChunker(min_tokens=80, max_tokens=500)
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        # Con agrupación, debería haber menos chunks que párrafos.
        assert len(chunks) < 5  # la nota tiene 5 párrafos

    def test_heading_path_vacio_sin_headings(self):
        parsed = parse_note(FIXTURES / "nota_corta_sin_headings.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        for c in chunks:
            assert c.heading_path == ""


# ---------------------------------------------------------------------------
# Índices y metadatos de chunks
# ---------------------------------------------------------------------------

class TestChunkMetadata:

    def test_chunk_index_secuencial(self):
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_path_correcto(self):
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        for c in chunks:
            assert c.source_path == "nota_larga_con_secciones.md"

    def test_token_count_positivo(self):
        parsed = parse_note(FIXTURES / "nota_larga_con_secciones.md")
        chunker = HybridChunker()
        chunks = chunker.chunk(parsed.body, parsed.metadata)
        for c in chunks:
            assert c.token_count > 0


# ---------------------------------------------------------------------------
# Casos borde
# ---------------------------------------------------------------------------

class TestCasosBorde:

    def test_body_vacio_devuelve_lista_vacia(self):
        chunker = HybridChunker()
        result = chunker.chunk("", _make_metadata())
        assert result == []

    def test_body_solo_whitespace(self):
        chunker = HybridChunker()
        result = chunker.chunk("   \n\n   \n", _make_metadata())
        assert result == []

    def test_body_una_linea(self):
        chunker = HybridChunker()
        result = chunker.chunk("Solo una línea corta.", _make_metadata())
        assert len(result) == 1
        assert result[0].text == "Solo una línea corta."


# ---------------------------------------------------------------------------
# Funciones auxiliares internas
# ---------------------------------------------------------------------------

class TestSplitByHeadings:

    def test_sin_headings_devuelve_vacio(self):
        assert _split_by_headings("Texto sin headings.") == []

    def test_con_headings(self):
        text = "## Intro\nContenido intro.\n\n## Final\nContenido final."
        sections = _split_by_headings(text)
        assert len(sections) == 2
        assert sections[0][0] == "## Intro"
        assert sections[1][0] == "## Final"

    def test_preamble_antes_de_heading(self):
        text = "Texto previo.\n\n## Heading\nContenido."
        sections = _split_by_headings(text)
        # El preamble es la primera sección (sin heading).
        assert sections[0][0] == ""
        assert "Texto previo" in sections[0][1]


class TestSplitByParagraphs:

    def test_divide_por_lineas_en_blanco(self):
        text = "Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres."
        result = _split_by_paragraphs(text)
        assert len(result) == 3

    def test_filtra_vacios(self):
        text = "Párrafo uno.\n\n\n\n\nPárrafo dos."
        result = _split_by_paragraphs(text)
        assert len(result) == 2


class TestSplitBySize:

    def test_texto_corto_un_chunk(self):
        result = _split_by_size("uno dos tres", max_tokens=100, overlap_tokens=10)
        assert len(result) == 1

    def test_texto_largo_varios_chunks(self):
        texto = " ".join(["palabra"] * 200)
        result = _split_by_size(texto, max_tokens=100, overlap_tokens=20)
        assert len(result) > 1

    def test_overlap_presente(self):
        """Los chunks consecutivos deben compartir algunas palabras (overlap)."""
        texto = " ".join(f"w{i}" for i in range(100))
        result = _split_by_size(texto, max_tokens=50, overlap_tokens=15)
        if len(result) >= 2:
            words_0 = set(result[0].split())
            words_1 = set(result[1].split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Debe haber overlap entre chunks consecutivos"

    def test_texto_vacio(self):
        assert _split_by_size("", max_tokens=100, overlap_tokens=10) == []


class TestGroupSmallParagraphs:

    def test_agrupa_parrafos_cortos(self):
        paragraphs = ["Corto.", "Muy corto.", "Otro corto."]
        result = _group_small_paragraphs(paragraphs, min_tokens=50)
        # Los tres se agrupan en uno solo.
        assert len(result) == 1

    def test_parrafo_grande_cierra_grupo(self):
        """Un párrafo grande que ya supera min_tokens cierra el grupo actual."""
        grande = " ".join(["palabra"] * 100)
        paragraphs = [grande, grande]
        result = _group_small_paragraphs(paragraphs, min_tokens=20)
        assert len(result) == 2

    def test_lista_vacia(self):
        assert _group_small_paragraphs([], min_tokens=50) == []
