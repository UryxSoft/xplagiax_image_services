"""
Gunicorn production configuration.

Key decisions:
  - workers=1: ONE Flask process to avoid loading models N times.
    ML models (CLIP ~1.5GB, SigLIP ~900MB) must not be duplicated.
    Heavy ML work is offloaded to the RQ worker process instead.

  - worker_class=gevent: async I/O via greenlets.
    Allows handling many concurrent HTTP connections (file uploads,
    status polling, search requests) without thread proliferation.
    ML inference blocks the greenlet but not other connections.

  - worker_connections=1000: max concurrent clients per worker.

  - timeout=120: generous timeout for large batch uploads.

  - For pure CPU scaling: increase workers (each loads its own model copy)
    only if you have sufficient RAM (budget ~10GB per worker).
    Alternatively, scale via multiple pods behind a load balancer,
    each with workers=1.
"""

import os
import multiprocessing

# --------------------------------------------------------------------------
# Server socket
# --------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "5004"))
bind = f"0.0.0.0:{PORT}"
backlog = 2048

# --------------------------------------------------------------------------
# Worker processes — see docstring above for reasoning
# --------------------------------------------------------------------------
workers = int(os.getenv("WORKER_PROCESSES", "1"))
worker_class = "gevent"
worker_connections = 1000

# --------------------------------------------------------------------------
# Timeouts
# --------------------------------------------------------------------------
timeout = 120           # 2 min for large uploads
graceful_timeout = 30   # time to finish in-flight requests on SIGTERM
keepalive = 5           # seconds to keep idle connections open

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
accesslog = "-"         # stdout — collected by Docker/k8s log driver
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = (
    '{"time":"%(t)s","method":"%(m)s","url":"%(U)s",'
    '"status":%(s)s,"bytes":%(B)s,"duration_ms":%(D)s,'
    '"remote":"%({X-Forwarded-For}i)s"}'
)

# --------------------------------------------------------------------------
# Process naming
# --------------------------------------------------------------------------
proc_name = "xplagiax_image_services"

# --------------------------------------------------------------------------
# Server mechanics
# --------------------------------------------------------------------------
preload_app = False     # False: each worker starts its own Flask app.
                        # True would share memory pre-fork (saves RAM) but
                        # breaks gevent workers. Keep False.

max_requests = 1000              # recycle worker after N requests (memory leak guard)
max_requests_jitter = 100        # stagger recycling to avoid all-at-once

# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------

# Hooks para logs con formato xplagiax
def on_starting(server):
    print(f"[xplagiax] Starting API server (Gunicorn + gevent)")
    print(f"[xplagiax] Port: {PORT} | Workers: {workers}")
    print(f"[xplagiax] Qdrant: {os.environ.get('QDRANT_HOST','qdrant')}:{os.environ.get('QDRANT_PORT','6333')}")
    print(f"[xplagiax] Redis:  {os.environ.get('REDIS_HOST','redis-server')}:{os.environ.get('REDIS_PORT','6379')}")
    print(f"[xplagiax] Storage: {os.environ.get('IMAGE_BACKEND','seaweedfs_filer')} → {os.environ.get('SEAWEEDFS_FILER_URL','')}")


def worker_exit(server, worker):
    server.log.info(f"Worker {worker.pid} exiting")


def post_fork(server, worker):
    """
    Called after each worker forks.
    Reinitialise random seeds and any fork-unsafe resources here.
    """
    import random
    import numpy as np
    random.seed()
    np.random.seed()
