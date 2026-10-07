# =============================================================================
# Tests — Evaluación de recall (sin Ollama/ChromaDB)
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_recall import load_queries, recall_at_k  # noqa: E402


class TestRecallAtK:

    def test_basico(self):
        assert recall_at_k(["a.md", "b.md"], ["a.md", "x.md"], 5) == 0.5

    def test_sin_hits(self):
        assert recall_at_k(["a.md"], ["x.md", "y.md"], 5) == 0.0

    def test_completo(self):
        assert recall_at_k(["a.md"], ["a.md"], 5) == 1.0

    def test_respeta_k(self):
        # b.md está en posición 3, fuera de top-2.
        assert recall_at_k(["a.md", "b.md"], ["a.md", "x.md", "b.md"], 2) == 0.5

    def test_sin_esperados(self):
        assert recall_at_k([], ["x.md"], 5) == 1.0


class TestLoadQueries:

    def test_carga_esqueleto_real(self, tmp_path):
        p = tmp_path / "q.yml"
        p.write_text(
            "- query: legumbres\n"
            "  n: 20\n"
            "  esperados:\n"
            "    - a.md\n"
            "    - b.md\n",
            encoding="utf-8",
        )
        cases = load_queries(p)
        assert len(cases) == 1
        assert cases[0]["query"] == "legumbres"
        assert cases[0]["n"] == 20
        assert cases[0]["esperados"] == ["a.md", "b.md"]

    def test_ignora_comentarios(self, tmp_path):
        p = tmp_path / "q.yml"
        p.write_text(
            "# comentario\n"
            "- query: zfs\n"
            "  n: 5\n"
            "  esperados:\n"
            "    - z.md\n",
            encoding="utf-8",
        )
        cases = load_queries(p)
        assert len(cases) == 1
        assert cases[0]["esperados"] == ["z.md"]
