# Rendimiento — optimizaciones aplicadas

Objetivo: reducir latencia/tiempo por imagen/lote **sin** tocar precisión, recall,
lógica de negocio ni el contrato de la API. Todas las optimizaciones de abajo
están **implementadas** y cubiertas por tests (82).

## Cuellos de botella (mapa)

| Camino | Coste dominante | Mitigación aplicada |
|---|---|---|
| Reverse search | **Red SerpApi 1–3 s + cuota $** | Caché de resultados + Session pooling + timeouts |
| Detección IA / búsqueda | **CPU (SigLIP > CLIP)** | Caché de resultados + CLIP∥SigLIP + `gthread` |
| Indexado | CPU (CLIP+SigLIP) | CLIP∥SigLIP concurrente |
| Validación/decyode | CPU decode | 2 aperturas (antes 3), cap de píxeles |

## Optimizaciones implementadas

| # | Optimización | Dónde | Efecto |
|---|---|---|---|
| 1 | **Caché de reverse search** (por hash; solo positivos → recall-safe) | `api_rotator.reverse_image_search`, `redis_client` | evita 1–3 s + cuota en repetidos |
| 2 | **Session HTTP con pool para SerpApi** | `api_rotator` (`_session`) | sin handshake TLS por llamada |
| 3 | **Caché de detección IA** (por hash, TTL 30d) | `blueprints.analyze_ai_detection` | evita re-inferir SigLIP |
| 5 | **CLIP ∥ SigLIP concurrente** (GIL liberado) | `registry.embed_and_classify` | inferencia `suma`→`máx` |
| 7 | **SHA-256 una sola vez** por request | `similarity`, `indexing`, `ml_worker` | menos CPU/hashing |
| 8 | **Timeouts/retries SerpApi** más agresivos | `config`, `api_rotator` | mejor p99 |
| 9 | **2 aperturas de imagen** (antes 3) | `image_validation` | 1 parse menos |
| 10 | **Embeddings en float32 bytes** (no JSON list) | `redis_client` | (de)serialización más barata, ~40% menos tamaño |
| — | **Worker gunicorn configurable** (gevent/gthread) | `gunicorn.conf.py` | concurrencia real para CPU |

> Precisión/recall intactos: la caché de reverse search **solo** guarda resultados
> positivos (un "no encontrado" siempre se reconsulta); la detección IA es
> determinista para los mismos bytes; no se redujo resolución ni se cambió el modelo.

## Reducción de latencia estimada

La variable crítica es la **tasa de repetición** del tráfico (activa las cachés).

| Escenario | ↓ Latencia media | Evidencia |
|---|---|---|
| Conservador | **15–25%** | Session pooling (~100 ms/call), hash-once, CLIP∥SigLIP (suma→máx, demostrable), `gthread`. Sin asumir cache-hits. |
| Realista | **40–60%** | + cachés con hit-rate ~30–50%: el path reverse pasa de ~1–3 s a ~5–20 ms en hits. |
| Agresivo | **70–90%** | Alta deduplicación: el path dominante (reverse) se sirve casi siempre de caché. |

## Pendiente (cambio de contrato — requiere decisión)

- Modo **async opcional** (`?async=true`) para `search`/`ai-detection` vía RQ, sin
  romper el modo síncrono actual.
- Batching cross-request real en el worker (agrupar jobs).
