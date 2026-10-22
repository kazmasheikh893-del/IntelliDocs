"""Embedding generation via the official google-genai SDK."""

import logging
from typing import List, Optional

import numpy as np
from django.conf import settings
from google import genai
from google.genai import types

from .errors import (  # noqa: F401  (re-exported for callers)
    EmbeddingError,
    MissingApiKeyError,
    translate_exception,
)

logger = logging.getLogger(__name__)

PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here"}
MAX_BATCH = 50


def get_api_key() -> str:
    key = (getattr(settings, "GEMINI_API_KEY", "") or "").strip()
    if key in PLACEHOLDER_KEYS:
        raise MissingApiKeyError()
    return key


def normalize(vec) -> List[float]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm == 0:
        raise EmbeddingError("The embedding model returned an unusable (zero or invalid) vector.")
    return (arr / norm).tolist()


class GeminiEmbeddingService:
    """Creates L2-normalised embeddings with a fixed model and dimensionality."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 dimensions: Optional[int] = None):
        key = (api_key or "").strip() or get_api_key()
        if key in PLACEHOLDER_KEYS:
            raise MissingApiKeyError()
        self.model = model or settings.GEMINI_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.GEMINI_EMBEDDING_DIM
        self._client = genai.Client(api_key=key)

    def embed_text(self, text: str) -> List[float]:
        """Embed a user query."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks (in API-sized batches)."""
        out: List[List[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            out.extend(self._embed(texts[i:i + MAX_BATCH], "RETRIEVAL_DOCUMENT"))
        return out

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        try:
            response = self._client.models.embed_content(
                model=self.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
        except Exception as exc:  # translated to a safe, credential-free error
            logger.warning("Embedding request failed: %s", type(exc).__name__)
            raise translate_exception(exc) from None

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise EmbeddingError("The embedding API returned an unexpected number of vectors.")
        vectors = []
        for emb in embeddings:
            values = list(emb.values or [])
            if len(values) != self.dimensions:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {self.dimensions}, got {len(values)}."
                )
            vectors.append(normalize(values))
        return vectors
