"""Optional human-in-the-loop blending.

The ATS score is in [0, 1] (cosine similarity). Human scores are typically
collected on a 1-10 Likert scale, so we normalize before blending.
"""

from __future__ import annotations

from typing import Dict


def normalize_human_score(human_score: float, scale_max: float = 10.0) -> float:
    """Map a human rating (default 1-10) into [0, 1]."""
    if scale_max <= 0:
        raise ValueError("scale_max must be positive.")
    normalized = human_score / scale_max
    return max(0.0, min(1.0, normalized))


def combine_scores(
    ats_score: float,
    human_score: float,
    alpha: float = 0.7,
    scale_max: float = 10.0,
) -> Dict[str, float]:
    """Blend ATS + human scores via `alpha`.

    final = alpha * ats + (1 - alpha) * human_normalized
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    human_normalized = normalize_human_score(human_score, scale_max=scale_max)
    final = alpha * ats_score + (1.0 - alpha) * human_normalized
    return {
        "ats_score": ats_score,
        "human_score_raw": human_score,
        "human_score_normalized": human_normalized,
        "alpha": alpha,
        "final_score": final,
    }
