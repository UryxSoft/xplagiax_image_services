XplagiaX Image Services — Paquete de cambios "cambios_1"
=========================================================

Contenido
---------
- cambios_1.patch          Parche unificado (git diff, binary-safe) de TODOS los cambios
                           respecto a origin/main.
- 0001-*.patch             Mismo cambio como commit (para `git am`).
- changed_files/           Copia de los archivos AÑADIDOS/MODIFICADOS en sus rutas
                           relativas al repo (por si prefieres copiarlos a mano).
- _changed.txt             Lista de archivos añadidos/modificados.
- _deleted.txt             Lista de archivos ELIMINADOS (ver más abajo).
- AUDIT_REPORT.md          Informe de auditoría + correcciones aplicadas.

Cómo aplicar (recomendado: el parche)
-------------------------------------
Desde la raíz del repo, con el árbol limpio en origin/main (o la rama que uses):

    # Opción A — aplicar como cambios al working tree:
    git apply --index cambios_1.patch

    # Opción B — aplicar como commit (conserva el mensaje):
    git am 0001-*.patch

Luego instala dependencias nuevas y corre los tests:

    pip install -r requirements.txt          # añade accelerate==0.33.0, imagehash pin
    python -m unittest tests.test_suite -v   # 71 tests

ELIMINACIONES IMPORTANTES (el parche ya las incluye; si copias a mano, BÓRRALOS)
-------------------------------------------------------------------------------
- .env                              (deja de versionarse; usa .env.example + secret manager)
- result.txt                        (logs de chat — fuga de info)
- test.py                           (script de pruebas suelto)
- app/storage/image_storage.py.bak  (backup obsoleto)
- .DS_Store, app/.DS_Store          (basura de macOS)

ACCIÓN MANUAL OBLIGATORIA (fuera del código)
--------------------------------------------
Las claves SerpApi/Zenserp que estaban hardcodeadas en app/config.py fueron
eliminadas del código, PERO siguen en el historial git. Debes:
  1) ROTAR/REVOCAR ya esas claves en los paneles de SerpApi y Zenserp.
  2) Purgar el historial (git filter-repo) si el repo es compartido.
  3) Inyectar las nuevas claves por variables de entorno / secret manager.
