"""
Model Registry — manages lifecycle of all ML models.

Key design decisions:
  1. LAZY LOADING: models are loaded after the Flask app starts,
     not at import time. A model load failure never crashes Flask.
  2. SINGLETON: one instance holds all loaded models.
     Multiple workers → each worker process has its own singleton,
     which is intentional: each process owns its GPU/CPU context.
  3. THREAD-SAFE: the load lock prevents duplicate loading on
     concurrent startup requests.
  4. GRACEFUL DEGRADATION: if SigLIP fails, embedding still works.
     If both fail, health endpoint returns 503 and explains why.
  5. torch.inference_mode(): preferred over no_grad() for inference —
     disables autograd engine entirely, ~5-10% faster.
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from requests.exceptions import Timeout

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, SiglipForImageClassification

from app.observability.telemetry import get_logger, get_metrics

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    vector: list[float]          # 512-dim normalized CLIP embedding
    model_id: str
    duration_ms: float


@dataclass
class ClassificationResult:
    is_ai: bool
    is_human: bool
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe model container with lazy initialization."""

    def __init__(
        self,
        siglip_model_id: str,
        clip_model_id: str,
        device: str,
        max_batch_size: int = 32,
    ) -> None:
        self._siglip_model_id = siglip_model_id
        self._clip_model_id = clip_model_id
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._max_batch_size = max_batch_size

        self._siglip: Optional[SiglipForImageClassification] = None
        self._processor: Optional[AutoImageProcessor] = None
        self._clip: Optional[SentenceTransformer] = None

        self._siglip_status = ModelStatus(
            name=siglip_model_id, loaded=False, device=device
        )
        self._clip_status = ModelStatus(
            name=clip_model_id, loaded=False, device=device
        )

        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load both models. Call from app startup hook, not from request path."""
        with self._load_lock:
            self._load_clip()
            self._load_siglip()

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        start = time.perf_counter()
        try:
            logger.info("loading_clip_model", model_id=self._clip_model_id)

            # SentenceTransformer relies on requests or huggingface_hub implicitly under the hood.
            # Using environment variables to set a timeout for the downloads:
            import os
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

            self._clip = SentenceTransformer(
                self._clip_model_id,
                device=str(self._device),
            )
            # Warm-up pass — first inference is slower due to JIT / CUDA init
            dummy = Image.new("RGB", (224, 224))
            self._clip.encode(dummy, normalize_embeddings=True, show_progress_bar=False)

            elapsed = time.perf_counter() - start
            self._clip_status = ModelStatus(
                name=self._clip_model_id,
                loaded=True,
                device=str(self._device),
                load_time_s=round(elapsed, 2),
            )
            get_metrics().model_loaded.labels(model_name="clip").set(1)
            logger.info("clip_model_loaded", elapsed_s=round(elapsed, 2))
        except Exception as exc:
            self._clip_status = ModelStatus(
                name=self._clip_model_id,
                loaded=False,
                device=str(self._device),
                error=str(exc),
            )
            get_metrics().model_loaded.labels(model_name="clip").set(0)
            logger.error("clip_model_load_failed", error=str(exc), exc_info=True)
            # Do NOT re-raise — degraded mode

    def _load_siglip(self) -> None:
        if self._siglip is not None:
            return
        start = time.perf_counter()
        try:
            logger.info("loading_siglip_model", model_id=self._siglip_model_id)

            import os
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

            self._processor = AutoImageProcessor.from_pretrained(self._siglip_model_id)
            self._siglip = SiglipForImageClassification.from_pretrained(
                self._siglip_model_id
            )
            self._siglip.to(self._device)
            self._siglip.eval()

            if str(self._device) == "cuda":
                self._siglip = self._siglip.half()   # FP16 on GPU for 2x speed

            # Warm-up pass
            dummy = Image.new("RGB", (224, 224))
            inputs = self._processor(images=dummy, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                self._siglip(**inputs)

            elapsed = time.perf_counter() - start
            self._siglip_status = ModelStatus(
                name=self._siglip_model_id,
                loaded=True,
                device=str(self._device),
                load_time_s=round(elapsed, 2),
            )
            get_metrics().model_loaded.labels(model_name="siglip").set(1)
            logger.info("siglip_model_loaded", elapsed_s=round(elapsed, 2))
        except Exception as exc:
            self._siglip_status = ModelStatus(
                name=self._siglip_model_id,
                loaded=False,
                device=str(self._device),
                error=str(exc),
            )
            get_metrics().model_loaded.labels(model_name="siglip").set(0)
            logger.error("siglip_model_load_failed", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> list[EmbeddingResult]:
        """
        Compute CLIP embeddings for a batch of PIL images.
        Batch size is capped at max_batch_size.

        Returns a list of EmbeddingResult in the same order as input.
        Raises RuntimeError if CLIP model is not loaded.
        """
        if self._clip is None:
            raise RuntimeError(
                "CLIP model is not loaded. "
                f"Load error: {self._clip_status.error}"
            )

        results = []
        batch_size = min(len(images), self._max_batch_size)

        # Process in sub-batches if needed
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            start = time.perf_counter()

            embeddings = self._clip.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalised for cosine similarity
                show_progress_bar=False,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            get_metrics().clip_inference_duration.labels(
                batch_size=str(len(batch))
            ).observe(elapsed_ms / 1000)

            for emb in embeddings:
                vec = emb.flatten().astype("float32")
                # Verify normalisation (should be ~1.0)
                norm = float(np.linalg.norm(vec))
                if norm < 1e-8:
                    raise ValueError(
                        "CLIP produced a near-zero embedding — image may be corrupt. "
                        "Refusing to index a zero-vector that would poison Qdrant."
                    )
                results.append(
                    EmbeddingResult(
                        vector=vec.tolist(),
                        model_id=self._clip_model_id,
                        duration_ms=elapsed_ms / len(batch),
                    )
                )

            del batch
            del embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "clip_inference_complete",
            images=len(images),
            avg_ms=round(elapsed_ms / len(images), 1),
        )
        return results

    @torch.inference_mode()
    def classify_ai_human(
        self, images: list[Image.Image]
    ) -> list[ClassificationResult]:
        """
        Run SigLIP batch inference. Returns one ClassificationResult per image.
        Raises RuntimeError if SigLIP is not loaded.
        """
        if self._siglip is None:
            raise RuntimeError(
                "SigLIP model is not loaded. "
                f"Load error: {self._siglip_status.error}"
            )

        results = []
        batch_size = min(len(images), self._max_batch_size)

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            start = time.perf_counter()

            inputs = self._processor(
                images=batch, return_tensors="pt"
            ).to(self._device)

            outputs = self._siglip(**inputs)
            probs = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

            elapsed_ms = (time.perf_counter() - start) * 1000
            get_metrics().siglip_inference_duration.labels(
                batch_size=str(len(batch))
            ).observe(elapsed_ms / 1000)

            id2label = self._siglip.config.id2label

            for prob_row in probs:
                scores = {
                    id2label[j]: float(prob_row[j])
                    for j in range(len(prob_row))
                }
                top_label = max(scores, key=scores.get)
                top_confidence = scores[top_label]

                # Normalise label variants ('AI', 'ai', 'artificial', etc.)
                is_ai = top_label.lower() in ("ai", "artificial", "generated", "fake")
                is_human = not is_ai

                ai_score = max(
                    (v for k, v in scores.items() if k.lower() in ("ai", "artificial", "generated", "fake")),
                    default=1.0 - top_confidence if is_human else top_confidence,
                )
                human_score = 1.0 - ai_score

                get_metrics().siglip_confidence.observe(top_confidence)

                results.append(
                    ClassificationResult(
                        is_ai=is_ai,
                        is_human=is_human,
                        label=top_label,
                        confidence=top_confidence,
                        ai_score=round(ai_score, 6),
                        human_score=round(human_score, 6),
                        all_scores={k: round(v, 6) for k, v in scores.items()},
                        model_id=self._siglip_model_id,
                        duration_ms=elapsed_ms / len(batch),
                    )
                )

            del batch
            del inputs
            del outputs
            del probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Single-image convenience wrappers
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
            },
            "siglip": {
                "model_id":    self._siglip_status.name,
                "loaded":      self._siglip_status.loaded,
                "device":      self._siglip_status.device,
                "load_time_s": self._siglip_status.load_time_s,
                "error":       self._siglip_status.error,
                "labels":      (
                    list(self._siglip.config.id2label.values())
                    if self._siglip else []
                ),
            },
            "device": str(self._device),
        }
