"""
gunicorn.conf.py — configuración mínima RAM
"""

import os

PORT = int(os.getenv("PORT", "5004"))
bind = f"0.0.0.0:{PORT}"
backlog = 512                    # reducido de 2048 — menos conexiones en cola

# Worker model is configurable:
#   - gevent  (default): best for the I/O-bound endpoints (reverse search, storage,
#     job polling). NOTE: CLIP/SigLIP inference is CPU-bound and blocks a gevent
#     worker while it runs (the GIL is held during native compute), so /search,
#     /plagiarism and /ai-detection do NOT get true concurrency under gevent.
#   - gthread: set GUNICORN_WORKER_CLASS=gthread (+ GUNICORN_THREADS) for
#     inference-heavy deployments — real OS threads let torch release the GIL.
# INFERENCE_MAX_CONCURRENCY still bounds simultaneous inferences to protect RAM.
workers = int(os.getenv("WEB_CONCURRENCY", os.getenv("WORKER_PROCESSES", "2")))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gevent")
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))   # used when worker_class=gthread

# Timeouts
timeout = 120
graceful_timeout = 30
keepalive = 2                    # reducido de 5

# Logging
accesslog = "-"
errorlog  = "-"
loglevel  = os.getenv("LOG_LEVEL", "warning").lower()  # warning > info en prod
access_log_format = '{"t":"%(t)s","m":"%(m)s","u":"%(U)s","s":%(s)s,"ms":%(D)s}'

proc_name   = "xplagiax"
preload_app = False

# Reciclar worker periódicamente para liberar memoria fragmentada
max_requests        = 500
max_requests_jitter = 50

# Límites de seguridad
limit_request_line         = 4094
limit_request_fields       = 50
limit_request_field_size   = 4094


def on_starting(server):
    print(f"[xplagiax] API en puerto {PORT} | worker=gevent | RAM-optimized")


def post_fork(server, worker):
    import random, numpy as np
    random.seed()
    np.random.seed()