"""
gunicorn.reverse_search.conf.py — Gunicorn config for the standalone
reverse-image-search microservice.

Worker class: gevent. Every request here is I/O-bound — SHA-256 hashing and
image-header validation cost microseconds; the dominant cost is waiting on
an external provider's HTTP response. gevent monkey-patches the socket layer
so plain, synchronous-looking `requests` calls actually yield to the event
loop under the hood, giving cooperative concurrency with zero async/await
code. Unlike the main xplagiax Dockerfile (where CLIP/SigLIP inference is
CPU-bound and holds the GIL, fighting gevent), there is no local inference
here at all, so gevent is the right choice with no caveats.
"""

import os

PORT = int(os.getenv("PORT", "5020"))
bind = f"0.0.0.0:{PORT}"
backlog = 512

workers = int(os.getenv("WEB_CONCURRENCY", "2"))
worker_class = "gevent"
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))

# Generous relative to the sum of per-provider timeouts + retries, but still bounded —
# a request can never hang the worker indefinitely.
timeout = 30
graceful_timeout = 15
keepalive = 2

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "warning").lower()
access_log_format = '{"t":"%(t)s","m":"%(m)s","u":"%(U)s","s":%(s)s,"ms":%(D)s}'

proc_name = "xplagiax-reverse-search"
preload_app = False

# Recycle workers periodically to release fragmented memory.
max_requests = 1000
max_requests_jitter = 100

limit_request_line = 4094
limit_request_fields = 50
limit_request_field_size = 4094


def on_starting(server):
    print(f"[reverse-search] listening on {PORT} | worker=gevent | no local ML")
