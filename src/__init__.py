# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# Buscador Semántico Local — Paquete principal
# =============================================================================
#
# Este paquete implementa un buscador semántico para notas en Markdown.
# Usa embeddings (vectores numéricos que representan el significado del texto)
# para encontrar contenido relevante por significado, no por palabras exactas.
#
# Componentes:
#   config.py              → Configuración (variables de entorno, constantes)
#   models.py              → Estructuras de datos (Chunk, NoteMetadata, etc.)
#   frontmatter_parser.py  → Extracción de metadatos YAML de los ficheros MD
#   chunker.py             → Troceado inteligente del texto para embeddings
#   embeddings.py          → Cliente HTTP para Ollama (genera los vectores)
#   store.py               → Operaciones con ChromaDB (base de datos vectorial)
#   indexer.py             → Orquestación de indexación (completa e incremental)
#   search.py              → Motor de búsqueda y formateo de resultados
#   cli.py                 → Interfaz de línea de comandos
# =============================================================================
