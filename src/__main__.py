# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# __main__.py — Permite ejecutar el paquete como módulo: python -m src
# =============================================================================
# Esto hace que `python -m src` equivalga a `python -m src.cli`,
# que es lo que usa el ENTRYPOINT del Dockerfile.
# =============================================================================

from src.cli import main

main()
