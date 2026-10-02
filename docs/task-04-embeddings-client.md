# Task 4: Cliente de embeddings para Ollama

## Estado: ✅ HECHO

## Objetivo

Implementar `embeddings.py` que se comunica con Ollama `/api/embed` para generar vectores. Soporte para batch (lotes), manejo de errores, reintentos con backoff exponencial, y prefijos específicos del modelo `nomic-embed-text`.

## Ficheros a crear

| Fichero | Descripción |
|---------|-------------|
| `src/embeddings.py` | Clase `OllamaEmbedder` con métodos para documentos y queries |
| `tests/test_embeddings.py` | Tests unitarios con mock de Ollama + test de integración |

## Detalle de implementación

### src/embeddings.py — OllamaEmbedder

**Constructor:**

```python
class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        # base_url: URL de Ollama (ej: "http://192.0.2.1:11434")
        # model: nombre del modelo (ej: "nomic-embed-text")
        # Crea cliente httpx con timeout configurado
```

**Métodos principales:**

`embed_documents(texts: list[str]) -> list[list[float]]`

1. Añade el prefijo `"search_document: "` a cada texto.
2. Divide la lista en lotes de `EMBED_BATCH_SIZE` (default 50).
3. Para cada lote, envía POST a `{base_url}/api/embed` con body:
   ```json
   {
     "model": "nomic-embed-text",
     "input": ["search_document: texto1", "search_document: texto2", ...]
   }
   ```
4. Ollama responde:
   ```json
   {
     "model": "nomic-embed-text",
     "embeddings": [[0.010, -0.001, ...], [0.023, 0.045, ...]]
   }
   ```
5. Concatena los vectores de todos los lotes y devuelve la lista completa.
6. Si un lote falla, reintenta con backoff exponencial (1s, 2s, 4s).

`embed_query(text: str) -> list[float]`

1. Añade el prefijo `"search_query: "` al texto.
2. Envía POST a `/api/embed` con un solo texto.
3. Devuelve el vector (lista de 768 floats para nomic-embed-text).

`health_check() -> bool`

1. Hace GET a `{base_url}/` (Ollama responde "Ollama is running" si está activo).
2. Devuelve True/False.

**Manejo de errores:**

- `httpx.ConnectError`: Ollama no es alcanzable → mensaje claro: "No se puede conectar a Ollama en {url}. ¿Está arrancado?"
- `httpx.TimeoutException`: timeout → reintento con backoff
- HTTP 404: modelo no encontrado → mensaje: "Modelo {model} no está disponible. Ejecuta: ollama pull {model}"
- Cualquier otro error HTTP → log del error y reintento

**Función de backoff:**

```python
def _request_with_retry(self, payload: dict) -> dict:
    for attempt in range(EMBED_RETRIES):
        try:
            response = self._client.post(url, json=payload, timeout=EMBED_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == EMBED_RETRIES - 1:
                raise
            wait = 2 ** attempt  # attempt=0 → 1s, attempt=1 → 2s, attempt=2 → 4s
            time.sleep(wait)
```

Nota: el backoff se aplica **tras** un fallo, no antes. El primer intento (attempt=0) se ejecuta inmediatamente. Si falla, se espera `2^0 = 1s` antes del segundo intento, `2^1 = 2s` antes del tercero, etc.

### Comentarios didácticos a incluir

- **Qué es un embedding**: Un vector de N números (768 para nomic-embed-text) que representa el significado del texto en un espacio matemático. Textos con significado similar producen vectores cercanos.

- **Prefijos search_document / search_query**: Particularidad de nomic-embed-text (y algunos otros modelos como E5). El modelo fue entrenado con estos prefijos para distinguir si el texto es un documento a indexar o una consulta de búsqueda. Esto mejora la calidad del matching. Sin los prefijos, los resultados son significativamente peores. NO todos los modelos necesitan prefijos — all-minilm no los necesita.

- **Por qué batch importa**: Enviar textos de uno en uno genera mucho overhead de red (HTTP request/response por cada texto). Con batch, enviamos 50 textos en una sola petición y Ollama devuelve 50 vectores. Para 2000 chunks, esto reduce de 2000 peticiones HTTP a 40.

- **Distancia coseno**: La métrica que usamos para comparar vectores. Mide el ángulo entre dos vectores en el espacio de N dimensiones. Valor de 0 = idénticos, valor de 2 = opuestos. No depende de la longitud de los vectores, solo de su dirección, lo cual es deseable para textos de diferente longitud.

- **Por qué el timeout es generoso**: En CPU (sin GPU), generar embeddings de un lote de 50 textos puede tardar 30-60 segundos dependiendo de la longitud. El timeout de 120s da margen para picos de carga o la primera petición (que tarda más porque Ollama carga el modelo en RAM).

### Tests

**Tests unitarios (con mock):**

- `test_embed_documents_anade_prefijo`: verificar que cada texto se envía con `"search_document: "` delante
- `test_embed_query_anade_prefijo`: verificar que la query se envía con `"search_query: "`
- `test_batch_divide_correctamente`: enviar 120 textos con batch_size=50 → genera 3 peticiones HTTP (50+50+20)
- `test_retry_en_timeout`: mock que falla las 2 primeras veces y responde la 3ª → devuelve resultado
- `test_error_modelo_no_encontrado`: mock 404 → excepción con mensaje claro
- `test_error_conexion`: mock ConnectionError → excepción con mensaje claro
- `test_health_check_ok`: mock 200 → True
- `test_health_check_falla`: mock ConnectionError → False

**Test de integración (skip si no hay Ollama):**

```python
@pytest.mark.integration
def test_embedding_real():
    embedder = OllamaEmbedder(base_url="http://...", model="nomic-embed-text")
    vector = embedder.embed_query("hola mundo")
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)
```
