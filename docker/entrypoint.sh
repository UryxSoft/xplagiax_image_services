#!/bin/sh
# ==============================================================
# entrypoint.sh — selects API or Worker mode at runtime
#
# Usage:
#   docker run xplagiax:latest          → API (gunicorn)
#   docker run xplagiax:latest worker   → RQ ML Worker
#   docker run xplagiax:latest shell    → bash (debug)
# ==============================================================

set -e

MODE="${1:-api}"

case "$MODE" in

  api)
    echo "[xplagiax] Starting API server (Gunicorn + gevent)"
    echo "[xplagiax] Port: ${PORT:-5004} | Workers: ${WORKER_PROCESSES:-1}"
    echo "[xplagiax] Qdrant: ${QDRANT_HOST}:${QDRANT_PORT}"
    echo "[xplagiax] Redis:  ${REDIS_HOST}:${REDIS_PORT}"
    echo "[xplagiax] Storage: ${IMAGE_BACKEND} → ${SEAWEEDFS_FILER_URL}"
    exec gunicorn \
      --config gunicorn.conf.py \
      "app.factory:create_app()"
    ;;

  worker)
    echo "[xplagiax] Starting ML Worker (RQ — queue: indexing)"
    echo "[xplagiax] Redis:  ${REDIS_HOST}:${REDIS_PORT}"
    echo "[xplagiax] Storage: ${IMAGE_BACKEND} → ${SEAWEEDFS_FILER_URL}"
    exec rq worker indexing \
      --url "redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB:-0}" \
      --max-jobs 0
    ;;

  shell)
    echo "[xplagiax] Starting shell (debug mode)"
    exec /bin/bash
    ;;

  *)
    echo "[xplagiax] ERROR: Unknown mode '${MODE}'"
    echo "Valid options: api | worker | shell"
    exit 1
    ;;

esac
