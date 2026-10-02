# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
"""check_no_pii.py — Bloquea PII y secretos en lo que se va a commitear.

Uso como hook pre-commit de git (bloquea el commit si encuentra algo)::

    scripts/install_hooks.sh   # una vez por clon

Uso manual::

    python3 scripts/check_no_pii.py              # revisa lo staged (igual que el hook)
    python3 scripts/check_no_pii.py --all        # audita todo el repo trackeado
    python3 scripts/check_no_pii.py fichero...  # revisa ficheros concretos

Qué detecta (todo en este fichero, sin dependencias):

- Claves privadas (``-----BEGIN ... PRIVATE KEY-----``).
- Tokens conocidos: AWS (``AKIA...``), GitHub (``ghp_...``/``github_pat_...``),
  estilo OpenAI (``sk-...`` con sufijo largo).
- Asignaciones sospechosas tipo ``API_KEY = "valor-largo"`` (el valor debe
  tener 12+ caracteres para no tragarse placeholders de tests).
- Emails reales (se permiten ``example.com`` / ``example.org`` / ``localhost``).
- Rutas absolutas locales con nombre de usuario real (``/Users/<alguien>``,
  ``/home/<alguien>``). El placeholder ``/Users/usuario`` está permitido.
- Lista personal del proyecto en ``scripts/pii_denylist.txt`` (literales,
  insensible a mayúsculas) más la lista local NO versionada
  ``scripts/pii_denylist.local.txt`` (ideal para valores reales: nunca se
  commitea, está en ``.gitignore``).

Códigos de salida: 0 = limpio, 1 = bloqueado (lista fichero:línea:motivo).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DENYLIST = ROOT / "scripts" / "pii_denylist.txt"
DENYLIST_LOCAL = ROOT / "scripts" / "pii_denylist.local.txt"
MAX_BYTES = 1_000_000  # no inspeccionar ficheros enormes (modelos, datasets)


# (nombre, regex compilada). Se evita a propósito palabras sueltas como
# "password" o "token": darían falsos positivos (ej: texto de la GPL,
# "estimate_tokens"). Solo patrones de alta confianza.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("clave privada",
     re.compile(r"-----BEGIN (?:RSA |OPENSSH |DSA |EC |PGP )?PRIVATE KEY-----")),
    ("AWS access key",
     re.compile(r"AKIA[0-9A-Z]{16}")),
    ("token de GitHub",
     re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("API key estilo OpenAI",
     re.compile(r"sk-(?:proj-|live-)?[A-Za-z0-9]{20,}")),
    ("asignación de secreto",
     re.compile(
         r"(?i)\b(api[_-]?key|api[_-]?secret|secret[_-]?key|client[_-]?secret|"
         r"auth[_-]?token|access[_-]?token|private[_-]?token|passwd|db[_-]?password)"
         r"\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+]{12,})['\"]?"
     )),
    ("email",
     re.compile(
         r"[A-Za-z0-9._%+-]+@(?!(?:example\.com|example\.org|localhost)\b)"
         r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
     )),
    ("ruta local /Users/<usuario>",
     re.compile(r"/Users/(?!usuario\b)[A-Za-z0-9_.-]+")),
    ("ruta local /home/<usuario>",
     re.compile(r"/home/(?!usuario\b)[A-Za-z0-9_.-]+")),
    ("ruta local Windows",
     re.compile(r"[A-Za-z]:\\Users\\(?!usuario\b)[A-Za-z0-9_.-]+")),
]


def load_denylist(path: Path) -> list[str]:
    """Lee literales a bloquear (ignora vacías y comentarios #)."""
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def staged_files() -> list[str]:
    """Ficheros staged (added/copied/modified). Lee el blob staged, no el disco."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACM"],
        capture_output=True, cwd=ROOT,
    )
    if out.returncode != 0:
        print("ERROR: no se pudo leer el staging de git.", file=sys.stderr)
        sys.exit(2)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def read_staged_blob(path: str) -> bytes | None:
    """Contenido staged de un fichero (None si no se puede leer)."""
    out = subprocess.run(
        ["git", "show", f":{path}"], capture_output=True, cwd=ROOT,
    )
    return out.stdout if out.returncode == 0 else None


def scan_text(rel: str, data: bytes, denylist: list[str]) -> list[str]:
    """Devuelve lista de 'fichero:línea: motivo' para un contenido."""
    if len(data) > MAX_BYTES or b"\x00" in data[:8000]:
        return []  # binario o demasiado grande: fuera de alcance
    findings = []
    text = data.decode("utf-8", errors="replace")
    lowered = text.lower()
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, rx in PATTERNS:
            if rx.search(line):
                findings.append(f"{rel}:{lineno}: posible {name}")
                break  # un aviso por línea basta
        else:
            line_lower = line.lower()
            for entry in denylist:
                if entry.lower() in line_lower:
                    findings.append(f"{rel}:{lineno}: coincide con denylist ({entry!r})")
                    break
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bloquea PII/secretos en commits.")
    parser.add_argument("files", nargs="*", help="ficheros a revisar (defecto: staged)")
    parser.add_argument("--all", action="store_true",
                        help="revisa todos los ficheros trackeados")
    args = parser.parse_args(argv)

    # La propia denylist contiene los literales: nunca escanearla.
    skip = {str(DENYLIST.relative_to(ROOT)), str(DENYLIST_LOCAL.relative_to(ROOT))}
    denylist = load_denylist(DENYLIST) + load_denylist(DENYLIST_LOCAL)

    targets: list[tuple[str, bytes]] = []
    if args.files:
        for f in args.files:
            p = Path(f)
            if p.is_file():
                targets.append((f, p.read_bytes()))
    elif args.all:
        out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, cwd=ROOT)
        for p in out.stdout.decode("utf-8").split("\0"):
            if p and p not in skip and (ROOT / p).is_file():
                targets.append((p, (ROOT / p).read_bytes()))
    else:
        for p in staged_files():
            if p in skip:
                continue
            blob = read_staged_blob(p)
            if blob is not None:
                targets.append((p, blob))

    findings: list[str] = []
    for rel, data in targets:
        findings.extend(scan_text(rel, data, denylist))

    if findings:
        print("BLOQUEADO por check_no_pii: posible PII o secreto detectado:")
        for f in findings:
            print(f"  {f}")
        print("Si es un falso positivo, ajusta los patrones o la denylist; "
              "para saltarlo una vez: git commit --no-verify")
        return 1
    print(f"check_no_pii OK ({len(targets)} fichero(s) revisados).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
