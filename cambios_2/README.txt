XplagiaX Image Services — Paquete de cambios "cambios_2" (ACUMULATIVO)
=====================================================================

IMPORTANTE: cambios_2 es ACUMULATIVO y REEMPLAZA a cambios_1.
Contiene TODO (los cambios de cambios_1 + las nuevas mejoras). Si vas a aplicar,
aplica SOLO este paquete; no apliques cambios_1 además.

Novedades respecto a cambios_1
------------------------------
- AI_LABEL_MAP: override de etiquetas (por índice o por string) para intercambiar
  modelos —incl. ultra-ligeros con etiquetas opacas— SIN tocar código.
- .github/workflows/ci.yml: CI que corre los 74 tests (con libs ML mockeadas) y un
  escaneo de vulnerabilidades de dependencias (pip-audit, informativo).
- .env.example: documenta el modelo por defecto, una alternativa ultra-ligera para
  CPU/poca RAM y el uso de AI_LABEL_MAP.
- docs/MODELS_AND_REVERSE_SEARCH.md: justificación del modelo de IA (CPU/poca RAM) y
  del método de reverse search (Google Lens), con límites honestos.

Contenido
---------
- cambios_2.patch          Parche unificado (binary-safe) de TODO vs origin/main.
- 0001-*.patch / 0002-*.patch   Los dos commits (para `git am`).
- changed_files/           Copia de los 23 archivos añadidos/modificados.
- _changed.txt / _deleted.txt   Listas de cambios y eliminaciones.
- AUDIT_REPORT.md, MODELS_AND_REVERSE_SEARCH.md

Cómo aplicar
------------
    git apply --index cambios_2.patch          # como cambios al working tree
    # o, como commits:
    git am 0001-*.patch 0002-*.patch

    pip install -r requirements.txt            # accelerate, imagehash pin, etc.
    python -m unittest tests.test_suite -v     # 74 tests

ELIMINACIONES (incluidas en el parche; si copias a mano, BÓRRALOS)
-----------------------------------------------------------------
.env  ·  result.txt  ·  test.py  ·  app/storage/image_storage.py.bak
.DS_Store  ·  app/.DS_Store

ACCIÓN MANUAL OBLIGATORIA
-------------------------
ROTAR/REVOCAR las claves SerpApi/Zenserp que estaban hardcodeadas (siguen en el
historial git). Purga el historial (git filter-repo) e inyecta las nuevas por
variables de entorno / secret manager.
