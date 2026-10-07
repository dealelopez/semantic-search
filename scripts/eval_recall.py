# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
"""eval_recall.py — Evaluación fija de recall@k (ESQUELETO, ver task-10).

Lee `scripts/eval_queries.yml` (query + ficheros esperados), ejecuta
`SearchEngine.search()` contra el índice real y reporta recall@k por query.

Uso::

    python3 scripts/eval_recall.py --queries scripts/eval_queries.yml
    python3 scripts/eval_recall.py --queries scripts/eval_queries.yml --json
    python3 scripts/eval_recall.py --queries scripts/eval_queries.yml -k 5

Necesita ChromaDB + Ollama arrancados (solo lee, no reindexa).
Exit code 0 siempre salvo error de conexión o fichero inválido.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permitir `python3 scripts/eval_recall.py` desde la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_queries(path: Path) -> list[dict]:
    """Carga el YAML de queries sin depender de PyYAML (stdlib only)."""
    import re

    text = path.read_text(encoding="utf-8")
    cases: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- query:"):
            if current is not None:
                cases.append(current)
            current = {
                "query": stripped.split(":", 1)[1].strip(),
                "n": 20,
                "esperados": [],
            }
        elif stripped.startswith("n:") and current is not None:
            current["n"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- ") and current is not None:
            # Entrada de lista bajo `esperados:`.
            current["esperados"].append(stripped[2:].strip())
        elif re.match(r"^esperados\s*:", stripped):
            continue
    if current is not None:
        cases.append(current)
    return cases


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Fracción de esperados presentes en los primeros k recuperados."""
    if not expected:
        return 1.0
    top_k = set(retrieved[:k])
    hits = sum(1 for e in expected if e in top_k)
    return hits / len(expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa recall@k del buscador semántico")
    parser.add_argument("--queries", default="scripts/eval_queries.yml")
    parser.add_argument("-k", type=int, default=None,
                        help="Override de n_results (si se omite, usa el `n` de cada caso)")
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args(argv)

    qpath = Path(args.queries)
    if not qpath.exists():
        print(f"ERROR: no existe {qpath}", file=sys.stderr)
        return 2
    cases = load_queries(qpath)
    if not cases:
        print(f"ERROR: {qpath} no contiene casos", file=sys.stderr)
        return 2

    # Import diferido para que --help y errores de fichero no requieran deps.
    from src.chunker import HybridChunker  # noqa: F401 (reservado para futura eval de chunking)
    from src.cli import _create_dependencies

    try:
        _embedder, _store, _chunker, _indexer, engine = _create_dependencies()
    except Exception as e:
        print(f"ERROR conectando (¿ChromaDB/Ollama arrancados?): {e}", file=sys.stderr)
        return 1

    rows = []
    for case in cases:
        n = args.k or case.get("n", 20)
        query = case["query"]
        expected = case.get("esperados", [])
        try:
            results, _elapsed = engine.search(query, n_results=n)
        except Exception as e:
            print(f"ERROR buscando {query!r}: {e}", file=sys.stderr)
            return 1
        retrieved = [r.source for r in results]
        scores = [r.score for r in results]
        r_at_n = recall_at_k(expected, retrieved, n)
        r_at_5 = recall_at_k(expected, retrieved, 5)
        missed = [e for e in expected if e not in retrieved[:n]]
        rows.append({
            "query": query,
            "n": n,
            "esperados": expected,
            "recall@5": round(r_at_5, 3),
            f"recall@{n}": round(r_at_n, 3),
            "hit": 1 if r_at_n > 0 else 0,
            "top1_score": scores[0] if scores else 0.0,
            "missed": missed,
            "top_sources": retrieved[:n],
        })

    if args.json:
        print(json.dumps({"results": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'query':<30}{'recall@5':<10}{'recall@n':<10}{'hit':<5}top1")
        print("─" * 70)
        for r in rows:
            rn_key = [k for k in r if k.startswith("recall@") and k != "recall@5"][0]
            print(f"{r['query']:<30}{r['recall@5']:<10}{r[rn_key]:<10}{r['hit']:<5}{r['top1_score']:.2f}")
        print("─" * 70)
        mean_r = sum(r["recall@5"] for r in rows) / len(rows)
        mean_hit = sum(r["hit"] for r in rows) / len(rows)
        print(f"MEDIA recall@5={mean_r:.2f} hit={mean_hit:.2f} ({len(rows)} casos)")
        for r in rows:
            if r["missed"]:
                print(f"  miss [{r['query']}]: {', '.join(r['missed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
