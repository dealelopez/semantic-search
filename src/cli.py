# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# cli.py — Punto de entrada del buscador semántico
# =============================================================================
#
# Interfaz de línea de comandos con subcomandos:
#
#   health     → Comprueba conexiones (ChromaDB, Ollama, modelo)
#   index      → Indexa notas (full o incremental)
#   search     → Busca por significado (one-shot o REPL interactivo)
#   status     → Muestra estadísticas de la colección
#   watch      → Vigila el directorio de notas y reindexa automáticamente
#   export     → Exporta el índice a JSON (backup)
#   import     → Importa un índice desde JSON (restore)
#   validate   → Valida la configuración (.env)
# =============================================================================

import argparse
import os
import readline  # noqa: F401 — habilita historial con flechas en input()
import sys
import threading
import time
from pathlib import Path

from rich.console import Console

from src import config
from src.chunker import HybridChunker
from src.embeddings import OllamaEmbedder, OllamaError
from src.indexer import Indexer
from src.search import SearchEngine, deduplicate_by_source
from src.store import VectorStore

console = Console()

# Fichero de historial del REPL interactivo.
_HISTORY_FILE = os.path.expanduser("~/.semantic_search_history")


# =============================================================================
# ARGPARSE — Definición de subcomandos y argumentos
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos con todos los subcomandos."""
    parser = argparse.ArgumentParser(
        prog="semantic-search",
        description="Buscador semántico local para notas Markdown",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Logging detallado (nivel DEBUG)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcomando a ejecutar")

    # --- health ---
    subparsers.add_parser("health", help="Comprueba conexiones con ChromaDB y Ollama")

    # --- index ---
    index_parser = subparsers.add_parser("index", help="Indexa las notas Markdown")
    index_parser.add_argument(
        "--mode",
        required=True,
        choices=["full", "incremental"],
        help="full = reindexar todo desde cero; incremental = solo cambios",
    )

    # --- search ---
    search_parser = subparsers.add_parser("search", help="Busca por significado")
    search_parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Texto de la consulta (requerido en modo one-shot)",
    )
    search_parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Modo REPL interactivo (búsquedas consecutivas)",
    )
    search_parser.add_argument(
        "--tags",
        default=None,
        help="Filtrar por tags, separados por coma (ej: ia,ml)",
    )
    search_parser.add_argument(
        "--type",
        default=None,
        dest="note_type",
        help="Filtrar por tipo de nota (ej: nota, proyecto, receta)",
    )
    search_parser.add_argument(
        "-n",
        type=int,
        default=config.DEFAULT_N_RESULTS,
        help=f"Número de resultados (default: {config.DEFAULT_N_RESULTS})",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Salida en formato JSON (para scripting)",
    )
    search_parser.add_argument(
        "--unique",
        action="store_true",
        help="Un resultado por nota (el de mayor similitud)",
    )
    search_parser.add_argument(
        "--explain",
        action="store_true",
        help="Muestra detalles de la búsqueda (filtros, tiempos, distancias)",
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Umbral mínimo de similitud (0.0-1.0). Descarta resultados por debajo",
    )
    search_parser.add_argument(
        "--keyword", "-k",
        default=None,
        help="Término exacto que debe aparecer en el texto (búsqueda híbrida)",
    )
    search_parser.add_argument(
        "--full",
        action="store_true",
        help="Muestra el texto completo de cada chunk (sin truncar)",
    )

    # --- status ---
    subparsers.add_parser("status", help="Muestra estadísticas de la colección")

    # --- watch ---
    watch_parser = subparsers.add_parser(
        "watch", help="Vigila el directorio de notas y reindexa automáticamente",
    )
    watch_parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Segundos de espera tras un cambio antes de reindexar (default: 3)",
    )

    # --- export ---
    export_parser = subparsers.add_parser("export", help="Exporta el índice a JSON (backup)")
    export_parser.add_argument(
        "filepath",
        help="Ruta del fichero JSON de destino",
    )

    # --- import ---
    import_parser = subparsers.add_parser(
        "import", help="Importa un índice desde JSON (restore)",
    )
    import_parser.add_argument(
        "filepath",
        help="Ruta del fichero JSON de origen",
    )

    # --- validate ---
    subparsers.add_parser("validate", help="Valida la configuración (.env)")

    return parser


# =============================================================================
# FACTORY — Creación de dependencias
# =============================================================================

def _create_dependencies():
    """Crea todas las dependencias del sistema a partir de la configuración."""
    embedder = OllamaEmbedder(
        base_url=config.OLLAMA_BASE_URL,
        model=config.EMBEDDING_MODEL,
    )
    store = VectorStore(
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        collection_name=config.COLLECTION_NAME,
    )
    chunker = HybridChunker()
    indexer = Indexer(embedder=embedder, store=store, chunker=chunker)
    search_engine = SearchEngine(embedder=embedder, store=store)
    return embedder, store, chunker, indexer, search_engine


# =============================================================================
# HANDLERS — Lógica de cada subcomando
# =============================================================================

def _cmd_health(embedder: OllamaEmbedder, store: VectorStore) -> None:
    """Subcomando health: comprueba conexiones."""
    console.print("\n[bold]Comprobando conexiones...[/bold]")
    all_ok = True

    # ChromaDB
    try:
        stats = store.collection_stats()
        console.print(
            f"  [green]OK[/green] ChromaDB ({config.CHROMA_HOST}:{config.CHROMA_PORT}) "
            f"— conectado ({stats['total_chunks']} chunks indexados)"
        )
    except Exception as e:
        console.print(
            f"  [red]ERROR[/red] ChromaDB ({config.CHROMA_HOST}:{config.CHROMA_PORT}) "
            f"— no se puede conectar"
        )
        console.print(f"     Error: {e}")
        all_ok = False

    # Ollama
    try:
        if embedder.health_check():
            if embedder.model_check():
                console.print(
                    f"  [green]OK[/green] Ollama ({config.OLLAMA_BASE_URL}) "
                    f"— conectado, modelo {config.EMBEDDING_MODEL} disponible"
                )
            else:
                console.print(
                    f"  [yellow]WARN[/yellow] Ollama ({config.OLLAMA_BASE_URL}) "
                    f"— conectado, pero modelo {config.EMBEDDING_MODEL} no disponible"
                )
                console.print(
                    f"     Ejecuta: ollama pull {config.EMBEDDING_MODEL}"
                )
                all_ok = False
        else:
            console.print(
                f"  [red]ERROR[/red] Ollama ({config.OLLAMA_BASE_URL}) — no responde"
            )
            all_ok = False
    except Exception as e:
        console.print(
            f"  [red]ERROR[/red] Ollama ({config.OLLAMA_BASE_URL}) — no se puede conectar"
        )
        console.print(f"     Error: {e}")
        all_ok = False

    console.print()
    if all_ok:
        console.print("  [green]Todo OK. Listo para indexar y buscar.[/green]")
    else:
        console.print("  [yellow]WARN[/yellow] Revisa las conexiones antes de continuar.")
    console.print()


def _cmd_index(indexer: Indexer, mode: str) -> None:
    """Subcomando index: indexa las notas."""
    notes_dir = config.NOTES_DIR

    # Aviso temprano si no hay nada que indexar. Además de informar, evita
    # que un `full` borre la colección (reset) por un directorio mal montado.
    if not Path(notes_dir).is_dir() or not any(Path(notes_dir).rglob("*.md")):
        console.print(
            f"  [yellow]WARN[/yellow] No se encontraron ficheros .md en {notes_dir}. "
            "¿Has montado las notas?"
        )
        return

    if mode == "full":
        console.print("\n[bold]Indexación COMPLETA[/bold] — se reindexará todo desde cero")
        console.print(f"   Directorio: {notes_dir}")
        console.print()
        report = indexer.full_reindex(notes_dir)
    else:
        console.print("\n[bold]Indexación INCREMENTAL[/bold] — solo cambios")
        console.print(f"   Directorio: {notes_dir}")
        console.print()
        report = indexer.incremental_index(notes_dir)

    # Mostrar reporte.
    console.print()
    console.print("[bold]Reporte:[/bold]")
    console.print(f"   Notas procesadas: {report.notes_processed}")
    if mode == "incremental":
        console.print(f"   Notas sin cambios: {report.notes_skipped}")
        console.print(f"   Notas eliminadas:  {report.notes_deleted}")
    console.print(f"   Chunks creados:   {report.chunks_created:,}")
    console.print(f"   Errores:          {len(report.errors)}")

    mins = int(report.duration_seconds) // 60
    secs = int(report.duration_seconds) % 60
    if mins > 0:
        console.print(f"   Duración:         {mins}m {secs}s")
    else:
        console.print(f"   Duración:         {secs}s")

    if report.errors:
        console.print()
        console.print(f"[yellow]WARN[/yellow] Errores ({len(report.errors)}):")
        for err in report.errors:
            console.print(f"   - {err}")
    console.print()


def _cmd_search_oneshot(
    search_engine: SearchEngine,
    query: str,
    n_results: int,
    filter_tags: list[str] | None,
    filter_type: str | None,
    json_output: bool = False,
    unique: bool = False,
    explain: bool = False,
    min_score: float = 0.0,
    keyword: str | None = None,
    full_text: bool = False,
) -> None:
    """Subcomando search en modo one-shot."""
    results, elapsed = search_engine.search(
        query, n_results, filter_tags, filter_type,
        keyword=keyword, min_score=min_score,
    )
    if unique:
        results = deduplicate_by_source(results)
    if json_output:
        search_engine.format_results_json(results, elapsed)
    elif explain:
        search_engine.format_results_explain(query, results, elapsed, filter_type, filter_tags)
    else:
        max_preview = 100_000 if full_text else None
        search_engine.format_results(query, results, elapsed, max_preview=max_preview)


def _cmd_search_interactive(
    search_engine: SearchEngine,
    n_results: int,
    filter_tags: list[str] | None,
    filter_type: str | None,
    unique: bool = False,
    min_score: float = 0.0,
    keyword: str | None = None,
) -> None:
    """Subcomando search en modo REPL interactivo con historial persistente."""
    # Cargar historial de búsquedas anteriores.
    try:
        readline.read_history_file(_HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass

    console.print("\n[bold]Buscador semántico — modo interactivo[/bold]")
    console.print("   Escribe tu consulta y pulsa Enter. 'q' o Ctrl+C para salir.")
    console.print("   /help para ver comandos disponibles.")
    console.print("━" * 60)

    state: dict = {
        "n_results": n_results,
        "filter_tags": filter_tags,
        "filter_type": filter_type,
        "unique": unique,
        "min_score": min_score,
        "keyword": keyword,
        "last_results": [],
    }

    while True:
        try:
            query = console.input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n   ¡Hasta luego!")
            break

        if not query:
            continue

        if query.lower() == "q":
            console.print("   ¡Hasta luego!")
            break

        if query.startswith("/"):
            _handle_repl_command(query, state, search_engine)
            continue

        # Ejecutar búsqueda.
        results, elapsed = search_engine.search(
            query,
            state["n_results"],
            state["filter_tags"],
            state["filter_type"],
            keyword=state["keyword"],
            min_score=state["min_score"],
        )
        if state["unique"]:
            results = deduplicate_by_source(results)
        state["last_results"] = results
        search_engine.format_results(query, results, elapsed)

    # Guardar historial al salir.
    try:
        readline.write_history_file(_HISTORY_FILE)
    except OSError:
        pass


def _handle_repl_command(command: str, state: dict, search_engine: SearchEngine) -> None:
    """Procesa un comando REPL (empieza con /)."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/tags":
        if arg:
            state["filter_tags"] = [t.strip() for t in arg.split(",") if t.strip()]
            console.print(f"   Filtro de tags activado: {', '.join(state['filter_tags'])}")
        else:
            state["filter_tags"] = None
            console.print("   Filtro de tags desactivado")

    elif cmd == "/type":
        if arg:
            state["filter_type"] = arg
            console.print(f"   Filtro de tipo activado: {state['filter_type']}")
        else:
            state["filter_type"] = None
            console.print("   Filtro de tipo desactivado")

    elif cmd == "/n":
        try:
            state["n_results"] = int(arg)
            console.print(f"   Resultados por búsqueda: {state['n_results']}")
        except ValueError:
            console.print("   [yellow]WARN[/yellow] Uso: /n 10")

    elif cmd == "/keyword":
        if arg:
            state["keyword"] = arg
            console.print(f"   Filtro keyword activado: \"{state['keyword']}\"")
        else:
            state["keyword"] = None
            console.print("   Filtro keyword desactivado")

    elif cmd == "/min-score":
        try:
            state["min_score"] = float(arg)
            console.print(f"   Score mínimo: {state['min_score']}")
        except ValueError:
            console.print("   [yellow]WARN[/yellow] Uso: /min-score 0.5")

    elif cmd == "/show":
        _handle_show(arg, state)

    elif cmd == "/unique":
        state["unique"] = not state.get("unique", False)
        status = "activado" if state["unique"] else "desactivado"
        console.print(f"   Deduplicación por fuente: {status}")

    elif cmd == "/clear":
        state["filter_tags"] = None
        state["filter_type"] = None
        state["unique"] = False
        state["keyword"] = None
        state["min_score"] = 0.0
        console.print("   Filtros limpiados")

    elif cmd == "/status":
        try:
            stats = search_engine.store.collection_stats()
            console.print(f"   Chunks indexados: {stats['total_chunks']:,}")
            console.print(f"   Notas indexadas:  {stats['total_sources']}")
        except Exception as e:
            console.print(f"   [red]ERROR[/red] Error: {e}")

    elif cmd == "/help":
        console.print("   Comandos disponibles:")
        console.print("     /tags tag1,tag2    — Filtrar por tags (vacío para quitar)")
        console.print("     /type tipo         — Filtrar por tipo de nota (vacío para quitar)")
        console.print("     /keyword término   — Búsqueda híbrida: exigir término exacto")
        console.print("     /min-score 0.5     — Umbral mínimo de similitud")
        console.print("     /n N               — Cambiar número de resultados")
        console.print("     /unique            — Alternar un resultado por nota")
        console.print("     /show N            — Ver texto completo del resultado N")
        console.print("     /clear             — Limpiar filtros")
        console.print("     /status            — Mostrar estado de la colección")
        console.print("     /help              — Mostrar esta ayuda")
        console.print("     q                  — Salir")

    else:
        console.print(f"   [yellow]WARN[/yellow] Comando no reconocido: {cmd}. Escribe /help")


def _handle_show(arg: str, state: dict) -> None:
    """Muestra el texto completo de un resultado de la última búsqueda."""
    last = state.get("last_results", [])
    if not last:
        console.print("   [yellow]WARN[/yellow] No hay resultados previos. Haz una búsqueda primero.")
        return
    try:
        idx = int(arg) - 1
    except ValueError:
        console.print(f"   [yellow]WARN[/yellow] Uso: /show 1  (números del 1 al {len(last)})")
        return
    if not (0 <= idx < len(last)):
        console.print(f"   [yellow]WARN[/yellow] Resultado {arg} no existe (hay {len(last)})")
        return
    r = last[idx]
    console.print()
    header = f"{r.source}"
    if r.heading:
        header += f" > {r.heading}"
    console.print(f"[bold]{header}[/bold]")
    console.print(f"   Score: {r.score:.4f}  |  Tipo: {r.note_type}  |  Tags: {', '.join(r.tags)}")
    console.print()
    console.print(r.text)
    console.print()


def _cmd_status(store: VectorStore) -> None:
    """Subcomando status: muestra estadísticas de la colección."""
    try:
        stats = store.collection_stats()
    except Exception as e:
        console.print(f"\n[red]ERROR[/red] No se puede conectar a ChromaDB: {e}")
        return

    console.print(f"\n[bold]Estado de la colección \"{stats['collection_name']}\"[/bold]")
    console.print("━" * 34)
    console.print(f"   Chunks indexados:  {stats['total_chunks']:,}")
    console.print(f"   Notas indexadas:   {stats['total_sources']}")
    console.print(f"   Colección:         {stats['collection_name']}")
    console.print(f"   ChromaDB:          {config.CHROMA_HOST}:{config.CHROMA_PORT}")
    console.print(f"   Ollama:            {config.OLLAMA_BASE_URL}")
    console.print(f"   Modelo:            {config.EMBEDDING_MODEL}")
    console.print()


def _cmd_watch(indexer: Indexer, delay: float) -> None:
    """
    Subcomando watch: vigila el directorio de notas y reindexanota automáticamente.

    Usa watchdog para detectar cambios en ficheros .md y lanza una
    indexación incremental tras un período de debounce (evita reindexar
    en cada keystroke del editor).
    """
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    notes_dir = config.NOTES_DIR

    console.print(f"\n[bold]Vigilando cambios en:[/bold] {notes_dir}")
    console.print(f"   Delay de debounce: {delay}s")
    console.print("   Ctrl+C para detener.")
    console.print("━" * 50)

    timer: threading.Timer | None = None
    lock = threading.Lock()

    def _do_incremental() -> None:
        console.print("\nCambio detectado — reindexando...")
        try:
            report = indexer.incremental_index(notes_dir, show_progress=False)
            console.print(
                f"   [green]OK[/green] {report.notes_processed} procesadas, "
                f"{report.notes_skipped} sin cambios, "
                f"{report.chunks_created} chunks en {report.duration_seconds:.1f}s"
            )
        except Exception as e:
            console.print(f"   [red]ERROR[/red] Error: {e}")

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):  # type: ignore[override]
            if event.is_directory:
                return
            src = getattr(event, "src_path", "")
            if not src.endswith(".md"):
                return
            nonlocal timer
            with lock:
                if timer is not None:
                    timer.cancel()
                timer = threading.Timer(delay, _do_incremental)
                timer.start()

    observer = Observer()
    observer.schedule(_Handler(), notes_dir, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n\n   Detenido. ¡Hasta luego!")
    finally:
        observer.stop()
        observer.join()
        with lock:
            if timer is not None:
                timer.cancel()


def _cmd_export(store: VectorStore, filepath: str) -> None:
    """Subcomando export: backup del índice a JSON."""
    console.print(f"\nExportando índice a: {filepath}")
    count = store.export_collection(filepath)
    console.print(f"   [green]OK[/green] {count:,} chunks exportados")
    console.print()


def _cmd_import(store: VectorStore, filepath: str) -> None:
    """Subcomando import: restaurar índice desde JSON."""
    from pathlib import Path
    if not Path(filepath).exists():
        console.print(f"\n[red]ERROR[/red] Fichero no encontrado: {filepath}")
        return
    console.print(f"\nImportando índice desde: {filepath}")
    console.print("   [yellow]WARN[/yellow] Esto reemplazará el índice actual.")
    count = store.import_collection(filepath, reset_first=True)
    console.print(f"   [green]OK[/green] {count:,} chunks importados")
    console.print()


def _cmd_validate() -> None:
    """Subcomando validate: comprueba la configuración."""
    console.print("\n[bold]Validando configuración...[/bold]")
    issues = config.validate_config()
    if not issues:
        console.print("  [green]OK[/green] Configuración correcta")
    else:
        console.print(f"  [yellow]WARN[/yellow] {len(issues)} problema(s) encontrado(s):")
        for issue in issues:
            console.print(f"   - {issue}")
    console.print()


# =============================================================================
# MAIN — Punto de entrada
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    """
    Punto de entrada principal de la CLI.

    Args:
        argv: Lista de argumentos (None = sys.argv). Se puede pasar
              explícitamente para testing.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configurar logging antes de cualquier otra cosa.
    config.setup_logging(verbose=args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # validate no necesita conexiones externas.
    if args.command == "validate":
        _cmd_validate()
        return

    # Crear dependencias.
    try:
        embedder, store, _chunker, indexer, search_engine = _create_dependencies()
    except Exception as e:
        console.print(f"\n[red]ERROR[/red] Error al conectar con los servicios: {e}")
        console.print("   ¿Están arrancados ChromaDB y Ollama?")
        console.print("   Ejecuta: docker compose up -d")
        sys.exit(1)

    # Dispatch al handler correspondiente.
    try:
        if args.command == "health":
            _cmd_health(embedder, store)

        elif args.command == "index":
            _cmd_index(indexer, args.mode)

        elif args.command == "search":
            filter_tags = None
            if args.tags:
                filter_tags = [t.strip() for t in args.tags.split(",") if t.strip()]

            if args.interactive:
                _cmd_search_interactive(
                    search_engine, args.n, filter_tags, args.note_type,
                    unique=args.unique,
                    min_score=args.min_score,
                    keyword=args.keyword,
                )
            elif args.query:
                _cmd_search_oneshot(
                    search_engine, args.query, args.n, filter_tags, args.note_type,
                    json_output=args.json, unique=args.unique, explain=args.explain,
                    min_score=args.min_score, keyword=args.keyword,
                    full_text=args.full,
                )
            else:
                console.print(
                    "[yellow]WARN[/yellow] Debes proporcionar una consulta o usar --interactive."
                )
                console.print("   Ejemplo: search \"mi consulta\"")
                console.print("   Ejemplo: search --interactive")
                sys.exit(1)

        elif args.command == "status":
            _cmd_status(store)

        elif args.command == "watch":
            _cmd_watch(indexer, args.delay)

        elif args.command == "export":
            _cmd_export(store, args.filepath)

        elif args.command == "import":
            _cmd_import(store, args.filepath)

    except OllamaError as e:
        console.print(f"\n[red]ERROR[/red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n\n   Interrumpido. ¡Hasta luego!")
        sys.exit(0)


if __name__ == "__main__":
    main()
