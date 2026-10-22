"""Vector storage and cosine-similarity retrieval (Django ORM + NumPy)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from documents.models import Document, DocumentChunk


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    document_title: str
    page_number: Optional[int]
    chunk_index: int
    content: str
    similarity_score: float


class BaseVectorStore(ABC):
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5,
               document_id: Optional[int] = None, min_similarity: float = 0.0,
               embedding_model: Optional[str] = None) -> List[SearchResult]:
        ...

    @abstractmethod
    def count_incompatible(self, embedding_model: str, dim: int,
                           document_id: Optional[int] = None) -> int:
        ...


def eligible_chunks(document_id: Optional[int] = None):
    """Chunks from successfully processed documents that actually have embeddings."""
    qs = DocumentChunk.objects.filter(document__status=Document.STATUS_READY).exclude(embedding=[])
    if document_id:
        qs = qs.filter(document_id=document_id)
    return qs


class ORMVectorStore(BaseVectorStore):
    """Embeddings live in a JSONField; cosine similarity is computed with NumPy."""

    def count_incompatible(self, embedding_model: str, dim: int,
                           document_id: Optional[int] = None) -> int:
        qs = eligible_chunks(document_id)
        compatible = qs.filter(embedding_model=embedding_model, embedding_dim=dim).count()
        return qs.count() - compatible

    def search(self, query_embedding: List[float], top_k: int = 5,
               document_id: Optional[int] = None, min_similarity: float = 0.0,
               embedding_model: Optional[str] = None) -> List[SearchResult]:
        if query_embedding is None or len(query_embedding) == 0 or top_k <= 0:
            return []
        try:
            q = np.asarray(query_embedding, dtype=np.float64)
        except (TypeError, ValueError):
            return []
        if q.ndim != 1 or not np.all(np.isfinite(q)):
            return []
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        qs = eligible_chunks(document_id).filter(embedding_dim=len(q))
        if embedding_model:
            qs = qs.filter(embedding_model=embedding_model)
        chunks = list(qs.select_related("document"))
        if not chunks:
            return []

        # Defensive: skip any row whose stored vector is malformed.
        valid, rows = [], []
        for c in chunks:
            emb = c.embedding
            if isinstance(emb, list) and len(emb) == len(q):
                valid.append(c)
                rows.append(emb)
        if not valid:
            return []
        matrix = np.asarray(rows, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        finite = np.isfinite(norms) & (norms > 0)
        norms[~finite] = 1.0
        scores = (matrix / norms[:, None]) @ q
        scores[~finite] = -1.0

        order = np.argsort(-scores)
        results: List[SearchResult] = []
        for idx in order:
            score = float(scores[idx])
            if score < min_similarity:
                break
            c = valid[idx]
            results.append(SearchResult(
                chunk_id=c.id, document_id=c.document_id, document_title=c.document.title,
                page_number=c.page_number, chunk_index=c.chunk_index,
                content=c.content, similarity_score=score,
            ))
            if len(results) >= top_k:
                break
        return results
