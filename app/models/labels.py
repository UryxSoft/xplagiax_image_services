"""
Pure label-semantics resolution for AI-vs-human classifiers.

Kept dependency-free (no torch/transformers) so it is cheap to import and easy
to unit-test. The registry maps a model's arbitrary class labels onto the
canonical {ai, human} space using these heuristics, and FAILS LOUD on unknown
labels instead of silently mislabelling images.
"""

from __future__ import annotations

from typing import Optional

_AI_LABEL_HINTS = (
    "ai", "artificial", "generated", "gen", "fake", "synthetic",
    "gan", "diffusion", "machine", "spoof", "deepfake",
)
_HUMAN_LABEL_HINTS = (
    "hum", "human", "real", "authentic", "natural", "photo",
    "camera", "genuine", "pristine",
)


def _classify_label(label: str) -> Optional[str]:
    """Return 'ai', 'human', or None (unknown) for a single class label."""
    norm = label.strip().lower().replace("-", " ").replace("_", " ")
    tokens = set(norm.split())

    human_hit = bool(tokens & set(_HUMAN_LABEL_HINTS)) or any(h in norm for h in _HUMAN_LABEL_HINTS)
    ai_hit = bool(tokens & set(_AI_LABEL_HINTS)) or any(h in norm for h in _AI_LABEL_HINTS)

    # Prefer the unambiguous case; if a label hits human hints and is not the
    # literal 'ai' token, treat as human.
    if human_hit and not (tokens & {"ai"} or norm.startswith("ai ")):
        return "human"
    if ai_hit:
        return "ai"
    if human_hit:
        return "human"
    return None


def resolve_label_semantics(id2label: dict, override: Optional[dict] = None) -> dict:
    """
    Map every class index to 'ai' or 'human'.

    `override` lets operators force the mapping for models whose labels the
    heuristics don't recognise (e.g. an ultra-light model with labels
    "0"/"1" or "class_a"/"class_b"). Keys may be the class index (int or str)
    or the label string (case-insensitive); values must be 'ai' or 'human'.

    Raises ValueError if any label cannot be classified, or if both classes
    are not represented.
    """
    norm_override = {}
    for k, v in (override or {}).items():
        val = str(v).strip().lower()
        if val not in ("ai", "human"):
            raise ValueError(f"AI_LABEL_MAP value must be 'ai' or 'human', got {v!r}")
        norm_override[str(k).strip().lower()] = val

    mapping: dict = {}
    for idx, label in id2label.items():
        label_s = str(label)
        sem = (
            norm_override.get(str(idx).strip().lower())
            or norm_override.get(label_s.strip().lower())
            or _classify_label(label_s)
        )
        if sem is None:
            raise ValueError(
                f"Unrecognised AI-detector label {label!r}. Set AI_LABEL_MAP "
                "(e.g. {\"0\":\"ai\",\"1\":\"human\"}) or use a model whose labels "
                "mention ai/human (e.g. 'ai'/'hum')."
            )
        mapping[int(idx)] = sem
    if "ai" not in mapping.values() or "human" not in mapping.values():
        raise ValueError(
            f"Model labels {list(id2label.values())} do not cover both ai and human."
        )
    return mapping
