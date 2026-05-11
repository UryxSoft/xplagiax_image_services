# ==============================================================
# xplagiax — Dockerfile ULTRA-LIGERO
# Objetivo: mínimo RAM, mínimo disco, mínimo CPU en idle
#
# Ahorros vs versión original:
#   ~1.5 GB imagen  → torch CPU-only wheel
#   ~800 MB RAM     → quantización INT8 de modelos
#   ~200 MB imagen  → limpieza agresiva de cache pip/HF
#   ~30% CPU idle   → límites de threads torch
# ==============================================================

# --------------------------------------------------------------
# Stage 1 — Builder
# --------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Deps de compilación — sólo los imprescindibles
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-light.txt .

# CLAVE: torch CPU-only desde índice oficial → ahorra ~1.5 GB
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --prefix=/install --no-cache-dir \
        -r requirements-light.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    # Eliminar archivos innecesarios de torch para reducir tamaño
    find /install -type f -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -path "*/torch/test*" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -name "*.pyx" -delete 2>/dev/null || true && \
    find /install -name "*.pyd" -delete 2>/dev/null || true


# --------------------------------------------------------------
# Stage 2 — Runtime
# --------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="xplagiax" \
      description="xplagiax Ultra-Light — CPU optimized" \
      version="2.0.0-light"

# Sólo runtime libs — sin libtiff ni libwebp si no los usas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root
RUN groupadd -r xplagiax && \
    useradd -r -g xplagiax -d /app -s /sbin/nologin xplagiax

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/         ./app/
COPY docker/gunicorn.conf.py ./gunicorn.conf.py

# ----------------------------------------------------------
# Pre-descarga de modelos con QUANTIZACIÓN INT8
# Esto reduce RAM en ~50% por modelo en CPU
# ----------------------------------------------------------
ARG DOWNLOAD_MODELS=true
ARG HF_HOME=/app/.cache/huggingface

ENV HF_HOME=${HF_HOME} \
    TRANSFORMERS_CACHE=${HF_HOME}

COPY docker/download_models_light.py /tmp/download_models.py

RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        echo "Descargando modelos con optimización CPU..." && \
        python /tmp/download_models.py && \
        # Limpiar cache de descarga de HuggingFace (blobs temporales)
        find ${HF_HOME} -name "*.lock" -delete && \
        find ${HF_HOME} -name "tmp*" -type f -delete; \
    fi

# Directorios y permisos
RUN mkdir -p \
        /app/.cache/huggingface \
        /tmp/xplagiax \
    && chown -R xplagiax:xplagiax /app /tmp/xplagiax

USER xplagiax

# ----------------------------------------------------------
# Variables de entorno — optimizadas para CPU/RAM
# ----------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    \
    # Flask
    FLASK_DEBUG=false \
    PORT=5004 \
    WORKER_PROCESSES=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    \
    # TORCH — limitar threads evita saturar CPU en idle
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    \
    # HuggingFace — deshabilitar telemetría y checks online
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    DISABLE_TQDM=1 \
    \
    # Qdrant
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=xplagiax_images \
    QDRANT_HNSW_M=16 \
    QDRANT_HNSW_EF_CONSTRUCT=200 \
    QDRANT_HNSW_EF_SEARCH=128 \
    \
    # Redis
    REDIS_HOST=redis-server \
    REDIS_PORT=6379 \
    REDIS_DB=0 \
    REDIS_SOCKET_TIMEOUT=1.0 \
    REDIS_EMBEDDING_TTL=86400 \
    REDIS_RESULT_TTL=300 \
    REDIS_JOB_TTL=3600 \
    \
    # SeaweedFS
    IMAGE_BACKEND=seaweedfs_filer \
    SEAWEEDFS_FILER_URL=http://seaweedfs-filer:8888 \
    SEAWEEDFS_PUBLIC_URL=http://localhost:8888 \
    SEAWEEDFS_REPLICATION=000 \
    SEAWEEDFS_COLLECTION=xplagiax \
    SEAWEEDFS_REQUEST_TIMEOUT=30.0 \
    SEAWEEDFS_MAX_RETRIES=3 \
    \
    # ML Models — versión ligera
    SIGLIP_MODEL_ID=Ateeqq/ai-vs-human-image-detector \
    CLIP_MODEL_ID=sentence-transformers/clip-ViT-B-32 \
    MODEL_DEVICE=cpu \
    MODEL_MAX_BATCH_SIZE=8 \
    \
    # Security
    REQUIRE_AUTH=true \
    MAX_IMAGE_BYTES=10485760 \
    ALLOWED_MIME_TYPES=jpeg,png,webp \
    RATE_LIMIT_PER_MINUTE=30 \
    RATE_LIMIT_PER_HOUR=500 \
    \
    # Observability — Prometheus desactivado por defecto (ahorra RAM)
    SERVICE_NAME=xplagiax \
    ENVIRONMENT=production \
    PROMETHEUS_ENABLED=false \
    PROMETHEUS_PORT=9090

EXPOSE 5004

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

COPY docker/entrypoint.sh ./entrypoint.sh
USER root
RUN chmod +x /app/entrypoint.sh && chown xplagiax:xplagiax /app/entrypoint.sh
USER xplagiax

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
