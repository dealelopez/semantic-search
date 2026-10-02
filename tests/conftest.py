# =============================================================================
# conftest.py — Configuración global de pytest
# =============================================================================
# ChromaDB usa pydantic-settings con extra='forbid', lo que significa que
# CUALQUIER variable de entorno no reconocida por ChromaDB causa un error
# al importar. Nuestras variables (.env) como OLLAMA_BASE_URL, CHROMA_HOST
# etc. conflictan con esto.
#
# Además, pydantic-settings lee el fichero .env del CWD automáticamente.
#
# Solución: limpiar las variables conflictivas Y parchear pydantic-settings
# para que no lea el .env al importar chromadb.
# =============================================================================

import os
import sys

# --- Paso 1: Limpiar variables de entorno ANTES de importar chromadb ---

_CONFLICTING_VARS = [
    "OLLAMA_BASE_URL", "EMBEDDING_MODEL", "CHROMA_HOST", "CHROMA_PORT",
    "COLLECTION_NAME", "NOTES_DIR", "LOG_LEVEL", "MAX_CHUNK_TOKENS",
    "MIN_CHUNK_TOKENS", "OVERLAP_TOKENS", "EMBED_BATCH_SIZE",
    "EMBED_TIMEOUT", "EMBED_RETRIES", "DEFAULT_N_RESULTS", "PREVIEW_MAX_CHARS",
]

_saved_env: dict[str, str] = {}
for _var in _CONFLICTING_VARS:
    if _var in os.environ:
        _saved_env[_var] = os.environ.pop(_var)

# --- Paso 2: Parchear pydantic-settings para ignorar .env durante import ---
# chromadb Settings hereda de BaseSettings que lee .env del CWD.
# Desactivamos la lectura de .env temporalmente forzando env_file vacío
# vía variable de entorno que pydantic NO recoge como setting de chromadb.

# La forma más robusta: monkeypatch DotEnvSettingsSource antes del import.
try:
    import pydantic_settings

    _orig_init = pydantic_settings.DotEnvSettingsSource.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["env_file"] = None
        _orig_init(self, *args, **kwargs)

    pydantic_settings.DotEnvSettingsSource.__init__ = _patched_init
except ImportError:
    pass
