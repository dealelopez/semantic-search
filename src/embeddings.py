# Copyright (C) 2026 Ale López
# SPDX-License-Identifier: GPL-3.0-or-later
# =============================================================================
# embeddings.py — Cliente HTTP para generar embeddings con Ollama
# =============================================================================
#
# Qué es un embedding:
# ---
# Un embedding es un vector de N números (768 para nomic-embed-text) que
# representa el "significado" de un texto en un espacio matemático. Textos
# con significado similar producen vectores cercanos en ese espacio.
#
# Ejemplo:
#   "El gato duerme"     → [0.23, -0.41, 0.87, ..., 0.12]  (768 números)
#   "El felino descansa" → [0.21, -0.39, 0.85, ..., 0.14]  (vectores cercanos!)
#   "Python es genial"   → [-0.55, 0.73, -0.12, ..., 0.66] (vector lejano)
#
# Prefijos search_document / search_query:
# ---
# nomic-embed-text (y otros modelos como E5) fue entrenado con prefijos
# especiales para distinguir si el texto es un documento a indexar o una
# consulta de búsqueda. Esto mejora significativamente la calidad del
# matching porque el modelo aprende que los documentos y las queries se
# formulan de formas diferentes.
#
# Sin los prefijos, los resultados de búsqueda son notablemente peores.
# NO todos los modelos necesitan prefijos — all-minilm no los necesita.
#
# Batching:
# ---
# Enviar textos de uno en uno genera mucho overhead de red (una petición
# HTTP por texto). Con batch, enviamos 50 textos en una sola petición y
# Ollama devuelve 50 vectores. Para 2000 chunks, esto reduce de 2000
# peticiones HTTP a 40. La diferencia en tiempo es enorme.
# =============================================================================

import logging
import time

import httpx

from src import config

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    Cliente HTTP para generar embeddings con Ollama.

    Usa la API /api/embed de Ollama, que acepta uno o varios textos
    y devuelve los vectores correspondientes.

    El cliente gestiona:
    - Prefijos del modelo (search_document / search_query)
    - Batching (dividir textos en lotes para no saturar Ollama)
    - Reintentos con backoff exponencial ante fallos transitorios
    - Timeouts generosos para hardware CPU-only
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: int | None = None,
        retries: int | None = None,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.EMBEDDING_MODEL
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self.timeout = timeout or config.EMBED_TIMEOUT
        self.retries = retries or config.EMBED_RETRIES

        # httpx client con timeout configurado.
        # No usamos un timeout global aquí — lo pasamos en cada petición
        # para poder diferenciarlo entre embed (lento) y health (rápido).
        self._client = httpx.Client(base_url=self.base_url)

    # =========================================================================
    # MÉTODOS PÚBLICOS
    # =========================================================================

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embeddings para una lista de textos (documentos a indexar).

        Añade el prefijo "search_document: " a cada texto antes de enviarlo.
        Divide la lista en lotes de EMBED_BATCH_SIZE para no saturar Ollama.

        Args:
            texts: Lista de textos a embeder.

        Returns:
            Lista de vectores (cada uno es una lista de floats de 768 dims).
            El orden corresponde al orden de los textos de entrada.

        Raises:
            OllamaError: Si Ollama no responde tras todos los reintentos.
        """
        if not texts:
            return []

        # Añadir prefijo de documento a cada texto.
        prefixed = [f"search_document: {t}" for t in texts]

        # Dividir en lotes y procesar.
        all_embeddings: list[list[float]] = []

        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(prefixed) + self.batch_size - 1) // self.batch_size

            logger.debug(
                "Embedding batch %d/%d (%d textos)",
                batch_num, total_batches, len(batch),
            )

            data = self._request_with_retry({"model": self.model, "input": batch})
            all_embeddings.extend(data["embeddings"])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Genera el embedding para una query de búsqueda.

        Añade el prefijo "search_query: " al texto. Este prefijo le dice
        al modelo que el texto es una consulta, no un documento. El modelo
        fue entrenado para que queries y documentos sobre el mismo tema
        produzcan vectores cercanos a pesar de estar formulados diferente.

        Args:
            text: Texto de la consulta.

        Returns:
            Vector de 768 floats.
        """
        prefixed = f"search_query: {text}"
        data = self._request_with_retry({"model": self.model, "input": [prefixed]})
        return data["embeddings"][0]

    def health_check(self) -> bool:
        """
        Verifica que Ollama está activo y responde.

        Hace GET a la raíz del servidor. Ollama responde con
        "Ollama is running" si está activo.

        Returns:
            True si Ollama responde, False si no.
        """
        try:
            response = self._client.get("/", timeout=10)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def model_check(self) -> bool:
        """
        Verifica que el modelo de embeddings está disponible en Ollama.

        Genera un embedding de prueba con un texto corto. Si funciona,
        el modelo está cargado y listo. Si falla con 404, el modelo
        no está instalado.

        Returns:
            True si el modelo responde correctamente, False si no.
        """
        try:
            self.embed_query("test")
            return True
        except OllamaError:
            return False

    def close(self) -> None:
        """Cierra el cliente HTTP y libera conexiones."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # =========================================================================
    # REINTENTOS CON BACKOFF EXPONENCIAL
    # =========================================================================

    def _request_with_retry(self, payload: dict) -> dict:
        """
        Envía una petición POST a /api/embed con reintentos automáticos.

        El backoff exponencial funciona así:
        - Intento 1: inmediato
        - Si falla → espera 2^0 = 1 segundo → intento 2
        - Si falla → espera 2^1 = 2 segundos → intento 3
        - Si falla → espera 2^2 = 4 segundos → intento 4 (si retries > 3)
        - Si todos fallan → lanza la excepción

        ¿Por qué backoff exponencial?
        Si Ollama está sobrecargado, bombardearlo con peticiones inmediatas
        empeora las cosas. Espaciar los reintentos le da tiempo para
        recuperarse. El factor exponencial hace que las esperas crezcan
        rápido, evitando recargar un servidor ya en problemas.

        En CPU, la primera petición tras arrancar Ollama suele tardar más
        porque el modelo se carga en RAM (~274MB). Los reintentos cubren
        este caso.
        """
        url = "/api/embed"
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                response = self._client.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                # Error 404 = modelo no encontrado → no reintentamos.
                if response.status_code == 404:
                    raise OllamaError(
                        f"Modelo '{self.model}' no disponible en Ollama. "
                        f"Ejecuta: ollama pull {self.model}"
                    )

                response.raise_for_status()
                return response.json()

            except httpx.ConnectError as e:
                raise OllamaError(
                    f"No se puede conectar a Ollama en {self.base_url}. "
                    f"¿Está arrancado el servidor? Error: {e}"
                ) from e

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Ollama: intento %d/%d falló (%s). Reintentando en %ds...",
                        attempt + 1, self.retries, type(e).__name__, wait,
                    )
                    time.sleep(wait)

        # Todos los reintentos agotados.
        raise OllamaError(
            f"Ollama no respondió tras {self.retries} intentos. "
            f"Último error: {last_error}"
        ) from last_error


# =============================================================================
# EXCEPCIÓN PERSONALIZADA
# =============================================================================

class OllamaError(Exception):
    """
    Error al comunicarse con Ollama.

    Envuelve los errores de httpx con mensajes más claros y accionables
    para el usuario. En lugar de un traceback críptico de HTTP, el usuario
    ve: "No se puede conectar a Ollama en http://... ¿Está arrancado?"
    """
    pass
