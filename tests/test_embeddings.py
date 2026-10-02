# =============================================================================
# Tests — Cliente de embeddings para Ollama
# =============================================================================
# Todos los tests unitarios usan mocks de httpx. No necesitan Ollama corriendo.
# El test de integración (@pytest.mark.integration) necesita un servidor real.
# =============================================================================

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.embeddings import OllamaEmbedder, OllamaError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_response(n_texts: int, dims: int = 768) -> dict:
    """Crea una respuesta simulada de Ollama /api/embed."""
    return {
        "model": "nomic-embed-text",
        "embeddings": [[0.1] * dims for _ in range(n_texts)],
    }


@pytest.fixture
def embedder():
    """Embedder con URL ficticia para tests unitarios."""
    return OllamaEmbedder(
        base_url="http://fake-ollama:11434",
        model="nomic-embed-text",
        batch_size=50,
        timeout=10,
        retries=3,
    )


# ---------------------------------------------------------------------------
# embed_documents — prefijos
# ---------------------------------------------------------------------------

class TestEmbedDocuments:

    def test_anade_prefijo_search_document(self, embedder):
        """Cada texto debe enviarse con 'search_document: ' delante."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_embed_response(2)

        with patch.object(embedder._client, "post", return_value=mock_response) as mock_post:
            embedder.embed_documents(["texto uno", "texto dos"])

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["input"] == [
                "search_document: texto uno",
                "search_document: texto dos",
            ]

    def test_lista_vacia_devuelve_vacia(self, embedder):
        result = embedder.embed_documents([])
        assert result == []

    def test_devuelve_vectores_correctos(self, embedder):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_embed_response(3)

        with patch.object(embedder._client, "post", return_value=mock_response):
            result = embedder.embed_documents(["a", "b", "c"])
            assert len(result) == 3
            assert len(result[0]) == 768


# ---------------------------------------------------------------------------
# embed_documents — batching
# ---------------------------------------------------------------------------

class TestBatching:

    def test_divide_en_lotes(self, embedder):
        """120 textos con batch_size=50 → 3 peticiones HTTP (50+50+20)."""
        embedder.batch_size = 50
        texts = [f"texto-{i}" for i in range(120)]

        call_count = 0
        def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            payload = kwargs.get("json", {})
            n = len(payload.get("input", []))
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _make_embed_response(n)
            return resp

        with patch.object(embedder._client, "post", side_effect=fake_post):
            result = embedder.embed_documents(texts)
            assert call_count == 3  # 50 + 50 + 20
            assert len(result) == 120

    def test_batch_unico_si_menos_que_batch_size(self, embedder):
        """10 textos con batch_size=50 → 1 sola petición."""
        embedder.batch_size = 50
        texts = [f"texto-{i}" for i in range(10)]

        call_count = 0
        def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _make_embed_response(10)
            return resp

        with patch.object(embedder._client, "post", side_effect=fake_post):
            embedder.embed_documents(texts)
            assert call_count == 1


# ---------------------------------------------------------------------------
# embed_query — prefijo
# ---------------------------------------------------------------------------

class TestEmbedQuery:

    def test_anade_prefijo_search_query(self, embedder):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_embed_response(1)

        with patch.object(embedder._client, "post", return_value=mock_response) as mock_post:
            embedder.embed_query("mi consulta")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["input"] == ["search_query: mi consulta"]

    def test_devuelve_un_solo_vector(self, embedder):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_embed_response(1)

        with patch.object(embedder._client, "post", return_value=mock_response):
            result = embedder.embed_query("test")
            assert isinstance(result, list)
            assert len(result) == 768


# ---------------------------------------------------------------------------
# Reintentos y errores
# ---------------------------------------------------------------------------

class TestRetryYErrores:

    def test_retry_en_timeout(self, embedder):
        """Falla 2 veces por timeout, responde la 3ª → éxito."""
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = _make_embed_response(1)

        call_count = 0
        def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("timeout")
            return success_response

        with patch.object(embedder._client, "post", side_effect=fake_post):
            with patch("src.embeddings.time.sleep"):  # no esperar de verdad
                result = embedder.embed_query("test")
                assert len(result) == 768
                assert call_count == 3

    def test_error_modelo_no_encontrado(self, embedder):
        """HTTP 404 → OllamaError con mensaje sobre el modelo."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch.object(embedder._client, "post", return_value=mock_response):
            with pytest.raises(OllamaError, match="no disponible"):
                embedder.embed_query("test")

    def test_error_conexion(self, embedder):
        """ConnectionError → OllamaError con mensaje sobre el servidor."""
        with patch.object(
            embedder._client, "post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(OllamaError, match="No se puede conectar"):
                embedder.embed_query("test")

    def test_reintentos_agotados(self, embedder):
        """Todos los reintentos fallan → OllamaError."""
        with patch.object(
            embedder._client, "post",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            with patch("src.embeddings.time.sleep"):
                with pytest.raises(OllamaError, match="no respondió"):
                    embedder.embed_query("test")


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_health_ok(self, embedder):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(embedder._client, "get", return_value=mock_response):
            assert embedder.health_check() is True

    def test_health_falla_conexion(self, embedder):
        with patch.object(
            embedder._client, "get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert embedder.health_check() is False

    def test_health_falla_timeout(self, embedder):
        with patch.object(
            embedder._client, "get",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            assert embedder.health_check() is False


# ---------------------------------------------------------------------------
# Test de integración (requiere Ollama real)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_embedding_real():
    """
    Test end-to-end con Ollama real. Skip si no hay servidor.
    Ejecutar con: pytest -m integration
    """
    embedder = OllamaEmbedder()
    try:
        if not embedder.health_check():
            pytest.skip("Ollama no disponible")
    except Exception:
        pytest.skip("Ollama no disponible")

    vector = embedder.embed_query("hola mundo")
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)
