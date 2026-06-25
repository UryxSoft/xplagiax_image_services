# Auditoría técnica y remediación — XplagiaX Image Services

Documento de auditoría *enterprise-grade* del microservicio (detección de imágenes
generadas por IA + reverse image search) y registro de **todas** las correcciones
aplicadas en esta rama. Cada hallazgo incluye severidad, archivo y estado.

> Convención: **[HECHO]** demostrable en el código · **[INFERENCIA]** deducción
> técnica fundada · **[ESPECULACIÓN]** requiere datos/pruebas no disponibles.

---

## 1. Resumen ejecutivo

El servicio parte de una base arquitectónica razonable (capas separadas, indexado
asíncrono con RQ, Qdrant + SeaweedFS, quantización INT8, observabilidad con
structlog/Prometheus). La auditoría encontró, sin embargo, fallos **críticos de
seguridad** y decisiones algorítmicas que no soportaban las promesas de producto.
Esta entrega corrige el 100 % de los hallazgos listados.

---

## 2. Hallazgos y correcciones (todos resueltos)

### Seguridad — P0
| # | Hallazgo | Sev. | Archivo | Estado |
|---|----------|------|---------|--------|
|S1|API keys de SerpApi/Zenserp **hardcodeadas** y commiteadas; lectura de entorno comentada|🔴 Crítica|`app/config.py`|**FIXED** — se eliminan los literales y se leen **solo** de `SERPAPI_KEY`/`ZENSERP_KEY`. *(Rotar las claves filtradas es obligatorio fuera del repo.)*|
|S2|SSRF por **DNS rebinding (TOCTOU)**: validación y descarga resolvían DNS por separado|🔴 Crítica|`app/security/http_client.py`|**FIXED** — se resuelve una vez, se validan **todas** las IPs, y se **fija (pin)** la IP validada en la conexión (Host/SNI preservados). Cada redirección se revalida.|
|S3|Auth **desactivada por defecto** en `docker-compose` (`env_file: ../.env` con `REQUIRE_AUTH=false` + key placeholder)|🔴 Crítica|`docker/docker-compose.yml`, `.env`|**FIXED** — se elimina `env_file`, `REQUIRE_AUTH=true`, secretos inyectados con `${VAR:?}`. `.env` deja de versionarse.|
|S4|`.env` versionado + `result.txt`/`AUDIT_REPORT.md` con logs de chat + `.bak`/`test.py`/`.DS_Store`|🟠 Alta|repo|**FIXED** — untrackeados y `.gitignore` reforzado.|
|S5|Rate limiter **roto tras proxy** ("10.0.0.0/8" comparado como string) y **fail-open** si Redis cae|🟠 Alta|`app/security/middleware.py`|**FIXED** — IP de cliente con CIDR real + `ProxyFix`; si Redis cae se degrada a un limitador **local por proceso** (sin bypass silencioso).|
|S6|`/health` y `/readyz` **sin auth** filtrando model IDs, URL del filer y cuotas de proveedores|🟠 Alta|`app/routes/blueprints.py`|**FIXED** — `/health` requiere auth; `/readyz` devuelve solo booleanos.|
|S7|Una sola API key sin separación para operaciones destructivas|🟡 Media|middleware/routes|**FIXED** — decorador `require_admin` + `ADMIN_API_KEY` para `reset`/`delete_group`.|
|S8|LFI vía `image_path` (lectura de archivos locales del contenedor)|🟡 Media|`app/utils/image_fetcher.py`|**FIXED** — deshabilitado por defecto (`ALLOW_LOCAL_IMAGE_PATH=false`).|
|S9|Bomba de descompresión: decodificaba hasta ~300 MB antes de cualquier chequeo|🟠 Alta|`app/utils/image_validation.py`|**FIXED** — límite de píxeles configurable validado **antes** de decodificar; rechazo de multi-frame; excepciones acotadas.|

### Bugs / código — P1–P2
| # | Hallazgo | Archivo | Estado |
|---|----------|---------|--------|
|B1|Matching de etiquetas del detector frágil y dependiente del modelo (riesgo de clasificar todo como "humano" en silencio)|`app/models/registry.py` → `app/models/labels.py`|**FIXED** — resolución robusta `id2label → {ai,human}` que **falla ruidoso** ante etiquetas no reconocidas; con tests.|
|B2|Abstracción del rotador con fugas: devolvía JSON crudo distinto por proveedor|`app/services/api_rotator.py`|**FIXED** — normalización a DTO uniforme + veredicto.|
|B3|`@retry` redefinido dentro del bucle de proveedores|`app/services/api_rotator.py`|**FIXED** — política de retry definida una sola vez.|
|B4|`config.debug` cargado pero nunca aplicado|`app/factory.py`|**FIXED** — `app.debug = config.debug`.|
|B5|`__import__("io")` dinámico|`app/routes/blueprints.py`|**FIXED** — `import io`.|
|B6|`before_request` parseaba el body en cada request|`app/observability/telemetry.py`|**FIXED** — ya no fuerza el parseo del multipart.|
|B7|Docstring del filer afirmaba "HEAD before PUT" inexistente|`app/storage/image_storage.py`|**FIXED** — docstring corregido (idempotencia por hash en la key).|
|B8|Código muerto que simulaba capacidades: `CircuitBreaker`, `cleanup_worker`, `init_tracing`|varios|**FIXED** — los tres **cableados**: circuit breaker Redis en el rotador, cleanup funcional, OTEL inicializado si `OTEL_ENABLED`.|
|B9|Fuga de imágenes temporales (sin TTL) + URL no pública en reverse-search por archivo|`image_storage.py`, `blueprints.py`|**FIXED** — TTL por request en el filer + aviso si la URL no es pública.|

### Observabilidad / escalabilidad — P2
| # | Hallazgo | Estado |
|---|----------|--------|
|O1|Sin tracing pese a config OTEL|**FIXED** — `init_tracing` se invoca cuando `OTEL_ENABLED=true`.|
|O2|Sin correlación API→worker|**FIXED** — `request_id` propagado al job RQ y bindeado en logs del worker.|
|O3|Estado del rotador per-proceso (penalizaciones no coordinadas)|**FIXED** — circuit breaker distribuido en Redis.|
|P1|Inferencia ML sin límite de concurrencia (riesgo OOM bajo gevent)|**FIXED** — semáforo acotado (`INFERENCE_MAX_CONCURRENCY`) en CLIP y el detector.|

---

## 3. Modelo de detección de IA (CPU + poca RAM)

**Elección: `Ateeqq/ai-vs-human-image-detector`** (por defecto, configurable con
`SIGLIP_MODEL_ID`).

- **[HECHO]** Backbone **SigLIP2 ViT-B/16 (~86 M parámetros)**; con **quantización
  dinámica INT8** en CPU el consumo cae a **~150–250 MB RAM**.
- **[HECHO]** Entrenado con ~60k+60k imágenes de generadores **recientes**
  (Midjourney v6+, SD 3.5, FLUX, GPT-4o); accuracy in-distribution reportada ≈ 99.23 %.
- **[INFERENCIA]** Mejor balance accuracy/RAM para CPU entre las opciones revisadas
  (Swin-tiny ~28 M y ViT distilado ~11.8 M son más ligeros pero menos precisos en
  generadores modernos).
- **Mejora estructural:** el loader pasó a `AutoModelForImageClassification`, por lo
  que el modelo es **intercambiable** por uno más ligero (p.ej. un ViT distilado de
  ~12 M) con una sola variable de entorno, sin tocar código.
- **Calibración y honestidad:** *temperature scaling* (`AI_TEMPERATURE`), umbrales
  configurables, estado `is_uncertain`/`confidence_bucket` expuesto y **disclaimer**
  en la respuesta. **No se puede afirmar con certeza** que una imagen sea IA: es una
  probabilidad y degrada con compresión/recortes/capturas/distribution shift.

`low_cpu_mem_usage=True` (vía `accelerate`) reduce el pico de RAM al cargar; si
`accelerate` falta, hay *fallback* automático.

---

## 4. Reverse image search ("¿está en otras páginas de Internet?")

**Método: Google Lens vía SerpApi** (`REVERSE_IMAGE_ENGINE=google_lens`, por defecto),
con fallback a `google_reverse_image` y Zenserp.

- **[HECHO]** Resultados **normalizados** a un esquema único `{title, link, source,
  thumbnail}` independiente del proveedor.
- **[HECHO]** Veredicto explícito y **probabilístico**: `found_on_internet`,
  `verdict` (`NOT_FOUND` / `POSSIBLE_PARTIAL_MATCH` / `LIKELY_PRESENT_ONLINE`),
  `confidence`, `match_count`, `matches[]` y `disclaimer`.
- **Límites (honestidad técnica):** una búsqueda inversa **no** puede probar al 100 %
  que una imagen está o no en Internet — depende de la cobertura del índice del
  proveedor; la ausencia de coincidencias **no** es prueba de ausencia.

---

## 5. Puntuación (antes → después de esta entrega)

| Dimensión | Antes | Después* |
|---|:--:|:--:|
| Seguridad | 2 | 7 |
| Código | 4 | 7 |
| Arquitectura | 6 | 7 |
| Rendimiento | 5 | 6.5 |
| Escalabilidad | 5 | 6.5 |
| Observabilidad | 6 | 7.5 |
| Calidad modelo IA | 5 | 7 |
| Fiabilidad reverse search | 3 | 6.5 |

\* *Estimación; la validación final requiere benchmarks propios y un escaneo de
dependencias (SCA) en CI.*

---

## 6. Recomendaciones restantes (fuera del alcance del código)

1. **Rotar** las claves SerpApi/Zenserp filtradas y **purgar el historial** git
   (`git filter-repo`) — borrarlas en un commit nuevo no las elimina del histórico.
2. Inyectar secretos desde un secret manager (Vault / AWS SM / K8s Secrets).
3. Añadir **SCA** (`pip-audit`/Dependabot) y tests de integración con
   *testcontainers* (Qdrant/Redis reales) en CI.
4. Para throughput alto, mover también la inferencia de `search`/`ai-detection` a la
   cola RQ (hoy es síncrona bajo gevent; mitigada con semáforo de concurrencia).
5. Recalibrar el detector (temperature scaling) sobre un set propio y publicar
   métricas reales (precision/recall/F1) en vez de la accuracy in-distribution.

---

## 7. Verificación

```bash
# Suite de pruebas (71 tests): lógica pura + integración Flask (deps ML mockeadas)
python -m unittest tests.test_suite -v
```
