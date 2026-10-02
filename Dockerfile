# =============================================================================
# Dockerfile — Buscador Semántico Local
# =============================================================================
# Imagen ligera de Python para ejecutar la CLI de búsqueda semántica.
# No incluye Ollama ni ChromaDB — esos corren en sus propios contenedores
# o servidores. Esta imagen solo contiene la aplicación Python.
# =============================================================================

FROM python:3.12-slim

# Evitar que Python genere ficheros .pyc y que bufferee stdout/stderr.
# PYTHONUNBUFFERED=1 es importante para ver logs en tiempo real en Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copiar e instalar dependencias primero (aprovecha la caché de capas de Docker).
# Si requirements.txt no cambia, Docker reutiliza esta capa sin reinstalar.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente de la aplicación.
COPY src/ ./src/

# Crear usuario sin privilegios para no ejecutar como root.
# Buena práctica de seguridad: si la app tiene una vulnerabilidad,
# el atacante no obtiene acceso root al contenedor.
RUN adduser --disabled-password --no-create-home --gecos "" appuser
USER appuser

# Punto de entrada por defecto: ejecutar la CLI.
# Se puede sobreescribir con: docker compose run app python -m src.cli <comando>
ENTRYPOINT ["python", "-m", "src.cli"]

# Comando por defecto si no se pasa ningún argumento: mostrar ayuda.
CMD ["--help"]
