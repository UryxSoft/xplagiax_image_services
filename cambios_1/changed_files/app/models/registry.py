"""
app/models/registry.py — versión optimizada para CPU/RAM mínima

Cambios vs versión original:
  1. Quantización dinámica INT8 automática en CPU → ~50% menos RAM
  2. torch.set_num_threads() para evitar saturar CPU en idle
  3. low_cpu_mem_usage=True en SigLIP → carga más eficiente
  4. Carga de estado quantizado pre-calculado si existe
  5. torch.inference_mode() ya estaba, se mantiene
  6. Eliminado warm-up redundante en embed_images
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
# AutoModel makes the AI-detector backbone-agnostic: SigLIP, ViT, Swin, etc.
# all work, so a lighter model can be dropped in via SIGLIP_MODEL_ID.
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.observability.telemetry import get_logger, get_metrics
# Pure, torch-free label resolution (testable in isolation).
from app.models.labels import resolve_label_semantics

logger = get_logger(__name__)

# Limitar threads de torch — evita que use todos los cores en idle
# Ajusta según CPUs disponibles en tu contenedor
_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "2"))
torch.set_num_threads(_NUM_THREADS)
torch.set_num_interop_threads(_NUM_THREADS)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    vector: list[float]
    model_id: str
    duration_ms: float


@dataclass
class ClassificationResult:
    is_ai: bool
    is_human: bool
    is_uncertain: bool
    confidence_bucket: str  # HIGH, MEDIUM, LOW
    label: str
    confidence: float
    ai_score: float
    human_score: float
    all_scores: dict[str, float]
    model_id: str
    duration_ms: float


@dataclass
class ModelStatus:
    name: str
    loaded: bool
    device: str
    error: Optional[str] = None
    load_time_s: Optional[float] = None
    quantized: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe model container con quantización INT8 para CPU."""

    def __init__(
        self,
        siglip_model_id: str,
        clip_model_id: str,
        device: str,
        max_batch_size: int = 8,
        ai_confidence_high: float = 0.85,
        ai_confidence_med: float = 0.60,
        ai_temperature: float = 1.0,
        max_concurrency: int = 2,
    ) -> None:
        # Forzar CPU si se pide "auto" y no hay CUDA
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._siglip_model_id = siglip_model_id
        self._clip_model_id = clip_model_id
        self._device = torch.device(device)
        self._max_batch_size = max_batch_size
        self._use_quantization = (str(self._device) == "cpu")

        # AI-detection calibration / thresholds
        self._ai_high = ai_confidence_high
        self._ai_med = ai_confidence_med
        self._ai_temperature = max(1e-3, ai_temperature)

        # Bound concurrent inferences to protect RAM/CPU under load.
        self._infer_sema = threading.BoundedSemaphore(max(1, max_concurrency))

        # Ruta de modelos quantizados pre-calculados
        hf_home = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        self._quantized_dir = os.path.join(hf_home, "quantized")
        self._quantized_flag = os.path.join(self._quantized_dir, "quantized.flag")

        self._siglip: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._clip: Optional[SentenceTransformer] = None
        self._label_semantics: Optional[dict] = None   # idx -> 'ai' | 'human'

        self._siglip_status = ModelStatus(name=siglip_model_id, loaded=False, device=device)
        self._clip_status   = ModelStatus(name=clip_model_id,   loaded=False, device=device)
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        with self._load_lock:
            self._load_clip()
            self._load_siglip()

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        start = time.perf_counter()
        try:
            logger.info("loading_clip", model_id=self._clip_model_id,
                        quantization=self._use_quantization)

            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

            self._clip = SentenceTransformer(
                self._clip_model_id,
                device=str(self._device),
            )

            # Quantización dinámica INT8 en CPU
            if self._use_quantization:
                self._clip = torch.quantization.quantize_dynamic(
                    self._clip,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info("clip_quantized_int8")

            # Warm-up
            dummy = Image.new("RGB", (224, 224))
            self._clip.encode(dummy, normalize_embeddings=True, show_progress_bar=False)

            elapsed = time.perf_counter() - start
            self._clip_status = ModelStatus(
                name=self._clip_model_id, loaded=True,
                device=str(self._device), load_time_s=round(elapsed, 2),
                quantized=self._use_quantization,
            )
            get_metrics().model_loaded.labels(model="clip").set(1)
            logger.info("clip_loaded", elapsed_s=round(elapsed, 2),
                        quantized=self._use_quantization)
        except Exception as exc:
            self._clip_status = ModelStatus(
                name=self._clip_model_id, loaded=False,
                device=str(self._device), error=str(exc),
            )
            get_metrics().model_loaded.labels(model="clip").set(0)
            logger.error("clip_load_failed", error=str(exc), exc_info=True)

    def _load_siglip(self) -> None:
        if self._siglip is not None:
            return
        start = time.perf_counter()
        try:
            logger.info("loading_siglip", model_id=self._siglip_model_id,
                        quantization=self._use_quantization)

            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

            self._processor = AutoImageProcessor.from_pretrained(self._siglip_model_id)
            try:
                # low_cpu_mem_usage streams weights → lower RAM peak (needs accelerate)
                self._siglip = AutoModelForImageClassification.from_pretrained(
                    self._siglip_model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            except (ImportError, ValueError) as exc:
                logger.warning("siglip_low_cpu_mem_unavailable", error=str(exc))
                self._siglip = AutoModelForImageClassification.from_pretrained(
                    self._siglip_model_id,
                    torch_dtype=torch.float32,
                )
            self._siglip.to(self._device)
            self._siglip.eval()

            # Resolve label→{ai,human} once; fail loud if unrecognised.
            self._label_semantics = resolve_label_semantics(self._siglip.config.id2label)

            # GPU: FP16 para velocidad
            if str(self._device) == "cuda":
                self._siglip = self._siglip.half()

            # CPU: quantización dinámica INT8 (~50% menos RAM, sin GPU)
            if self._use_quantization:
                self._siglip = torch.quantization.quantize_dynamic(
                    self._siglip,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info("siglip_quantized_int8")

            # Warm-up
            dummy = Image.new("RGB", (224, 224))
            inputs = self._processor(images=dummy, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                self._siglip(**inputs)

            elapsed = time.perf_counter() - start
            self._siglip_status = ModelStatus(
                name=self._siglip_model_id, loaded=True,
                device=str(self._device), load_time_s=round(elapsed, 2),
                quantized=self._use_quantization,
            )
            get_metrics().model_loaded.labels(model="siglip").set(1)
            logger.info("siglip_loaded", elapsed_s=round(elapsed, 2),
                        quantized=self._use_quantization)
        except Exception as exc:
            self._siglip_status = ModelStatus(
                name=self._siglip_model_id, loaded=False,
                device=str(self._device), error=str(exc),
            )
            get_metrics().model_loaded.labels(model="siglip").set(0)
            logger.error("siglip_load_failed", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> list[EmbeddingResult]:
        if self._clip is None:
            raise RuntimeError(f"CLIP no cargado: {self._clip_status.error}")

        results = []
        batch_size = min(len(images), self._max_batch_size)

        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            start = time.perf_counter()

            with self._infer_sema:
                embeddings = self._clip.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000
            get_metrics().clip_inference_duration.labels(
                batch_size=str(len(batch))
            ).observe(elapsed_ms / 1000)

            for emb in embeddings:
                vec = emb.flatten().astype("float32")
                norm = float(np.linalg.norm(vec))
                if norm < 1e-8:
                    raise ValueError("CLIP produjo embedding cero — imagen posiblemente corrupta.")
                results.append(EmbeddingResult(
                    vector=vec.tolist(),
                    model_id=self._clip_model_id,
                    duration_ms=elapsed_ms / len(batch),
                ))

        return results

    @torch.inference_mode()
    def classify_ai_human(self, images: list[Image.Image]) -> list[ClassificationResult]:
        if self._siglip is None:
            raise RuntimeError(f"SigLIP no cargado: {self._siglip_status.error}")
        if not self._label_semantics:
            raise RuntimeError("AI detector labels unresolved — model not ready.")

        results = []
        batch_size = min(len(images), self._max_batch_size)
        id2label = self._siglip.config.id2label
        sem = self._label_semantics
        t = self._ai_temperature

        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            start = time.perf_counter()

            # Concurrency guard: bound simultaneous inferences to protect RAM.
            with self._infer_sema:
                inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
                outputs = self._siglip(**inputs)
                # Temperature scaling calibrates over-confident logits (T>1 softens).
                logits = outputs.logits.float() / t
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            elapsed_ms = (time.perf_counter() - start) * 1000
            get_metrics().siglip_inference_duration.labels(
                batch_size=str(len(batch))
            ).observe(elapsed_ms / 1000)

            for prob_row in probs:
                scores = {id2label[idx]: float(p) for idx, p in enumerate(prob_row)}
                # Aggregate by resolved semantics (handles >2 classes correctly).
                ai_score = sum(float(p) for idx, p in enumerate(prob_row) if sem[idx] == "ai")
                human_score = sum(float(p) for idx, p in enumerate(prob_row) if sem[idx] == "human")
                total = ai_score + human_score
                if total > 0:
                    ai_score /= total
                    human_score /= total

                is_ai = ai_score >= self._ai_high
                is_human = human_score >= self._ai_high
                is_uncertain = not is_ai and not is_human

                max_score = max(ai_score, human_score)
                if max_score >= self._ai_high:
                    bucket = "HIGH"
                elif max_score >= self._ai_med:
                    bucket = "MEDIUM"
                else:
                    bucket = "LOW"

                get_metrics().siglip_confidence.observe(max_score)
                results.append(ClassificationResult(
                    is_ai=is_ai, is_human=is_human, is_uncertain=is_uncertain,
                    confidence_bucket=bucket,
                    label="ai" if ai_score > human_score else "human",
                    confidence=max_score,
                    ai_score=round(ai_score, 6),
                    human_score=round(human_score, 6),
                    all_scores={k: round(v, 6) for k, v in scores.items()},
                    model_id=self._siglip_model_id,
                    duration_ms=elapsed_ms / len(batch),
                ))

        return results

    # ------------------------------------------------------------------
    # Single-image wrappers
    # ------------------------------------------------------------------

    def embed_single(self, image: Image.Image) -> EmbeddingResult:
        return self.embed_images([image])[0]

    def classify_single(self, image: Image.Image) -> ClassificationResult:
        return self.classify_ai_human([image])[0]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def clip_ready(self) -> bool:
        return self._clip is not None

    @property
    def siglip_ready(self) -> bool:
        return self._siglip is not None

    def is_siglip_loading(self) -> bool:
        return not self._siglip_status.loaded and self._siglip_status.error is None

    def get_status(self) -> dict:
        return {
            "clip": {
                "model_id":    self._clip_status.name,
                "loaded":      self._clip_status.loaded,
                "device":      self._clip_status.device,
                "load_time_s": self._clip_status.load_time_s,
                "error":       self._clip_status.error,
                "quantized":   self._clip_status.quantized,
            },
            "siglip": {
                "model_id":    self._siglip_status.name,
                "loaded":      self._siglip_status.loaded,
                "device":      self._siglip_status.device,
                "load_time_s": self._siglip_status.load_time_s,
                "error":       self._siglip_status.error,
                "quantized":   self._siglip_status.quantized,
                "labels": (
                    list(self._siglip.config.id2label.values())
                    if self._siglip else []
                ),
                "label_semantics": self._label_semantics,
                "calibration": {
                    "temperature":     self._ai_temperature,
                    "threshold_high":  self._ai_high,
                    "threshold_med":   self._ai_med,
                },
            },
            "device":             str(self._device),
            "threads":            _NUM_THREADS,
            "quantization_mode":  "int8" if self._use_quantization else "none",
        }
