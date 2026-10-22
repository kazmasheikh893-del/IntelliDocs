"""
Embedding generation service using official Google Gemini API (google-genai SDK).
Model: text-embedding-004
"""

import time
from typing import List, Optional
from django.conf import settings
from google import genai
from google.genai.errors import APIError


class MissingApiKeyError(Exception):
    """Raised when GEMINI_API_KEY is empty or not configured."""
    pass


class EmbeddingError(Exception):
    """Raised when the embedding generation API request fails."""
    pass


class GeminiEmbeddingService:
    """Provides methods to generate vector embeddings using Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", "")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise MissingApiKeyError(
                "GEMINI_API_KEY is not configured. Please add your real Gemini API key to .env"
            )
        self.model = model or getattr(settings, "GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self._client = genai.Client(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        """Generates embedding for a single string."""
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
        batch_results = self.embed_batch([text])
        return batch_results[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates embeddings for a batch of strings, splitting into smaller chunks
        to respect API request payload limits.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=chunk,
                )
                if not response.embeddings:
                    raise EmbeddingError("Gemini API returned an empty embedding response.")

                for emb in response.embeddings:
                    all_embeddings.append(list(emb.values))

            except APIError as e:
                raise EmbeddingError(f"Google Gemini API error during embedding generation: {e.message} (code: {e.code})")
            except Exception as e:
                raise EmbeddingError(f"Unexpected error communicating with Gemini API: {str(e)}")

        return all_embeddings
