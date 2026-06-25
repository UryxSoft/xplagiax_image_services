# Detección de IA y Reverse Image Search — decisiones técnicas

## 1. Modelo de detección de IA (CPU + poca RAM)

**Por defecto: `Ateeqq/ai-vs-human-image-detector`** (configurable con `SIGLIP_MODEL_ID`).

| Criterio | Valor |
|---|---|
| Arquitectura | SigLIP2 ViT-B/16 (vision tower, ~86M parámetros) |
| RAM en CPU | ~150–250 MB con quantización dinámica INT8 |
| Entrenamiento | ~60k IA + 60k humanas; generadores recientes (MJ v6+, SD 3.5, FLUX, GPT-4o) |
| Accuracy (in-distribution) | ≈ 99.23% (test set del autor) |

**Por qué este modelo:** es el mejor equilibrio *accuracy/RAM* para CPU entre las
opciones revisadas. Alternativas más ligeras (Swin-tiny ~28M, ViT distilado ~12M)
bajan la RAM pero pierden precisión en generadores nuevos.

### Cómo es ahora (mejoras aplicadas)
- El loader usa `AutoModelForImageClassification` → **cualquier** clasificador HF
  (SigLIP, ViT, Swin…) es intercambiable con una sola variable de entorno.
- Resolución de etiquetas robusta `id2label → {ai, human}` que **falla ruidoso**
  si no reconoce las etiquetas (evita clasificar todo como "humano" en silencio).
- Override manual con `AI_LABEL_MAP` para modelos con etiquetas opacas:
  `AI_LABEL_MAP={"0":"human","1":"ai"}`.
- `low_cpu_mem_usage=True` (vía `accelerate`) + INT8 → menor pico de RAM.
- Calibración por *temperature scaling* (`AI_TEMPERATURE`), umbrales configurables,
  y la API expone `is_uncertain` / `confidence_bucket` + *disclaimer*.
- Semáforo de concurrencia (`INFERENCE_MAX_CONCURRENCY`) para proteger la RAM.

### Cambiar a un modelo ultra-ligero (<256 MB)
```bash
SIGLIP_MODEL_ID=jacoballessio/ai-image-detect-distilled   # ~12M params
AI_LABEL_MAP={"0":"human","1":"ai"}                       # si las etiquetas son opacas
```
No requiere cambios de código.

### Honestidad estadística
No se puede afirmar con **certeza** que una imagen sea generada por IA: es una
**probabilidad**. La accuracy ~99% es *in-distribution*; degrada con compresión
JPEG, recortes, capturas de pantalla, upscaling y generadores no vistos
(*distribution shift*). Reporta confianza calibrada + incertidumbre, no veredictos.

## 2. Reverse Image Search ("¿está en otras páginas de Internet?")

**Método: Google Lens vía SerpApi** (`REVERSE_IMAGE_ENGINE=google_lens`), con
fallback a `google_reverse_image` y a Zenserp.

- Resultados **normalizados** a un esquema único `{title, link, source, thumbnail}`
  independientemente del proveedor.
- Veredicto **probabilístico** explícito:
  `found_on_internet`, `verdict` (`NOT_FOUND` / `POSSIBLE_PARTIAL_MATCH` /
  `LIKELY_PRESENT_ONLINE`), `confidence`, `match_count`, `matches[]`, `disclaimer`.
- Circuit breaker distribuido (Redis) + reintentos con backoff + rotación de
  proveedores por score.

### Limitaciones (importantes)
Una búsqueda inversa **no** prueba al 100% que una imagen esté o no en Internet:
depende de la cobertura del índice del proveedor. La ausencia de coincidencias
**no** es prueba de ausencia. Para búsqueda por archivo subido,
`SEAWEEDFS_PUBLIC_URL` **debe** ser una URL pública (el proveedor la descarga);
los temporales usan `SEAWEEDFS_TTL` para auto-expirar.
