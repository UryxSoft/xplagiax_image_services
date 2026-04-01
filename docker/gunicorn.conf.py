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
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
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
proc_name = "xplagiax-api"

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

def on_starting(server):
    server.log.info("Gunicorn starting — xplagiax API")


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
