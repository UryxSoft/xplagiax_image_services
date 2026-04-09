"""
gunicorn.conf.py — configuración mínima RAM
"""

import os

PORT = int(os.getenv("PORT", "5004"))
bind = f"0.0.0.0:{PORT}"
backlog = 512                    # reducido de 2048 — menos conexiones en cola

# UN solo worker — los modelos ML (~1 GB) no se duplican
workers = 1
worker_class = "gevent"
worker_connections = 200         # reducido de 1000

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
    # Limitar threads por worker
    import torch
    n = int(os.getenv("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)