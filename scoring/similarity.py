"""Pluggable similarity scoring for ATS-style evaluation.

Two scorers are provided:

* `TfidfScorer` (default): bag-of-words overlap, fast, no model download.
  This is the closest open-source proxy to keyword-based ATS systems.
* `EmbeddingScorer`: sentence-transformers cosine similarity. Captures
  semantic overlap; heavier dependency, lazy-imported so it's optional.

Adding a new scorer is just subclassing `SimilarityScorer.score()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class SimilarityScorer(ABC):
    """Pairwise text similarity in the [0, 1] range."""

    @abstractmethod
    def score(self, text_a: str, text_b: str) -> float:
        raise NotImplementedError


class TfidfScorer(SimilarityScorer):
    """TF-IDF + cosine similarity. Lightweight and ATS-flavored."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TfidfScorer requires scikit-learn. Install with: "
                "pip install scikit-learn"
            ) from e
        self._ngram_range = ngram_range

    def score(self, text_a: str, text_b: str) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            ngram_range=self._ngram_range,
            stop_words="english",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(matrix[0], matrix[1])[0, 0]
        return float(sim)


class EmbeddingScorer(SimilarityScorer):
    """Sentence-transformer cosine similarity. Captures semantic overlap.

    Lazily imports `sentence_transformers` so the dependency is only required
    when this scorer is actually used.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "EmbeddingScorer requires sentence-transformers. Install with: "
                "pip install sentence-transformers"
            ) from e
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def score(self, text_a: str, text_b: str) -> float:
        import numpy as np

        emb = self._model.encode([text_a, text_b], normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        # Clamp to [0, 1] to keep semantics consistent with TF-IDF.
        return max(0.0, min(1.0, sim))


def build_scorer(name: str, *, embedding_model: Optional[str] = None) -> SimilarityScorer:
    """Factory so callers can pick a scorer by string from config/CLI."""
    key = name.strip().lower()
    if key == "tfidf":
        return TfidfScorer()
    if key in {"embedding", "embeddings"}:
        if embedding_model:
            return EmbeddingScorer(model_name=embedding_model)
        return EmbeddingScorer()
    raise ValueError(f"Unknown scorer '{name}'. Expected one of: tfidf, embedding.")


def compare(
    scorer: SimilarityScorer,
    original_resume: str,
    modified_resume: str,
    job_description: str,
) -> Dict[str, float]:
    """Score both resumes against the JD and return improvement delta."""
    original = scorer.score(original_resume, job_description)
    modified = scorer.score(modified_resume, job_description)
    return {
        "original_score": original,
        "modified_score": modified,
        "improvement": modified - original,
    }
