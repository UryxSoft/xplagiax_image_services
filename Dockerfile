# ==============================================================
# xplagiax — Unified Production Dockerfile
# Builds one image that runs as either API or ML Worker
# depending on the command passed at runtime.
#
# Build:
#   docker build -t xplagiax:latest .
#
# Run as API:
#   docker run --name xplagiax-api  ... xplagiax:latest
#   (uses default CMD → gunicorn)
#
# Run as Worker:
#   docker run --name xplagiax-worker ... xplagiax:latest worker
#   (overrides CMD → rq worker)
#
# Storage: SeaweedFS (plain HTTP — no boto3 needed)
# Vector DB: Qdrant
# Cache/Queue: Redis
# ==============================================================


# --------------------------------------------------------------
# Stage 1 — Builder
# Installs all Python dependencies into /install
# so the runtime stage stays lean and has no build tools
# --------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer cache — only invalidates if requirements change)
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# --------------------------------------------------------------
# Stage 2 — Runtime
# Minimal image: only runtime libs, no compilers
# --------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="xplagiax" \
      description="xplagiax — AI Image Detection, Similarity Search & SeaweedFS Storage" \
      version="1.0.0"

# Runtime-only system deps (shared libs for Pillow + curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    libwebp7 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Non-root user — never run as root in production
# ----------------------------------------------------------
RUN groupadd -r xplagiax && \
    useradd -r -g xplagiax -d /app -s /sbin/nologin -c "xplagiax service" xplagiax

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY app/         ./app/
COPY docker/gunicorn.conf.py ./gunicorn.conf.py

# ----------------------------------------------------------
# HuggingFace model pre-download (optional build arg)
#
# By default DOWNLOAD_MODELS=false — models are downloaded
# on first startup and cached in the HF_HOME volume.
#
# Set DOWNLOAD_MODELS=true to bake models INTO the image.
# This makes the image ~4GB larger but eliminates the cold
# start download on each new container.
#
# Build with models baked in:
#   docker build --build-arg DOWNLOAD_MODELS=true -t xplagiax:models .
# ----------------------------------------------------------
ARG DOWNLOAD_MODELS=true 
ARG HF_HOME=/app/.cache/huggingface

ENV HF_HOME=${HF_HOME} \
    TRANSFORMERS_CACHE=${HF_HOME}

RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        echo "Downloading HuggingFace models into image..." && \
        python -c " \
from transformers import AutoImageProcessor, SiglipForImageClassification; \
from sentence_transformers import SentenceTransformer; \
print('Downloading SigLIP...'); \
AutoImageProcessor.from_pretrained('Ateeqq/ai-vs-human-image-detector'); \
SiglipForImageClassification.from_pretrained('Ateeqq/ai-vs-human-image-detector'); \
print('Downloading CLIP...'); \
SentenceTransformer('clip-ViT-B-32'); \
print('All models downloaded.'); \
"; \
    else \
        echo "DOWNLOAD_MODELS=false — models will be downloaded on first start"; \
    fi

# ----------------------------------------------------------
# Directories and permissions
# ----------------------------------------------------------
RUN mkdir -p \
        /app/.cache/huggingface \
        /tmp/xplagiax \
    && chown -R xplagiax:xplagiax /app /tmp/xplagiax

USER xplagiax

# ----------------------------------------------------------
# Environment defaults
# All values can be overridden at runtime via -e or --env-file
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
    # SeaweedFS storage (default backend)
    IMAGE_BACKEND=seaweedfs_filer \
    SEAWEEDFS_FILER_URL=http://seaweedfs-filer:8888 \
    SEAWEEDFS_PUBLIC_URL=http://localhost:8888 \
    SEAWEEDFS_REPLICATION=000 \
    SEAWEEDFS_COLLECTION=xplagiax \
    SEAWEEDFS_REQUEST_TIMEOUT=30.0 \
    SEAWEEDFS_MAX_RETRIES=3 \
    \
    # ML Models
    SIGLIP_MODEL_ID=Ateeqq/ai-vs-human-image-detector \
    CLIP_MODEL_ID=clip-ViT-B-32 \
    MODEL_DEVICE=auto \
    MODEL_MAX_BATCH_SIZE=32 \
    \
    # Security
    REQUIRE_AUTH=true \
    MAX_IMAGE_BYTES=20971520 \
    ALLOWED_MIME_TYPES=jpeg,png,webp,bmp,tiff,gif \
    RATE_LIMIT_PER_MINUTE=30 \
    RATE_LIMIT_PER_HOUR=500 \
    \
    # Observability
    SERVICE_NAME=xplagiax \
    ENVIRONMENT=production \
    PROMETHEUS_ENABLED=true \
    PROMETHEUS_PORT=9090

# ----------------------------------------------------------
# Exposed ports
#   5004  → Flask API (Gunicorn)
#   9090  → Prometheus metrics
# ----------------------------------------------------------
EXPOSE 5004 9090

# ----------------------------------------------------------
# Healthcheck — liveness probe
# start_period=90s gives models time to load on cold start
# ----------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# ----------------------------------------------------------
# Entrypoint script — selects API or Worker mode
# Usage:
#   (no args)  → gunicorn API server
#   worker     → rq worker (ML inference)
#   shell      → bash (debug only)
# ----------------------------------------------------------
COPY docker/entrypoint.sh ./entrypoint.sh

# entrypoint.sh is copied as root then ownership is set
# The USER directive above applies, so we need to set exec bit
USER root
RUN chmod +x /app/entrypoint.sh && chown xplagiax:xplagiax /app/entrypoint.sh
USER xplagiax

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
