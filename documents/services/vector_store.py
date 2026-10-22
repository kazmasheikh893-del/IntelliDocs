"""Vector storage and cosine-similarity retrieval (Django ORM + NumPy)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set

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
               embedding_model: Optional[str] = None,
               page_number: Optional[int] = None) -> List[SearchResult]:
        ...

    @abstractmethod
    def count_incompatible(self, embedding_model: str, dim: int,
                           document_id: Optional[int] = None) -> int:
        ...

    @abstractmethod
    def get_document_chunks(self, document_id: int) -> List[SearchResult]:
        ...

    def expand_context_window(self, matches: List[SearchResult], max_window: int = 1) -> List[SearchResult]:
        return matches


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

    def get_document_chunks(self, document_id: int) -> List[SearchResult]:
        """Returns all chunks for a ready document ordered by chunk_index."""
        qs = DocumentChunk.objects.filter(
            document_id=document_id,
            document__status=Document.STATUS_READY
        ).select_related("document").order_by("chunk_index")
        
        return [
            SearchResult(
                chunk_id=c.id,
                document_id=c.document_id,
                document_title=c.document.title,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                content=c.content,
                similarity_score=1.0,
            )
            for c in qs
        ]

    def expand_context_window(self, matches: List[SearchResult], max_window: int = 1) -> List[SearchResult]:
        """
        Expands top matches with immediate neighboring chunks from the same document
        and same/adjacent page, avoiding fragmented definitions across chunk boundaries.
        Preserves original ranking while inserting context neighbors in natural order.
        """
        if not matches or max_window <= 0:
            return matches

        matched_doc_ids = {m.document_id for m in matches}
        expanded_chunks: List[SearchResult] = []
        seen_chunk_ids: Set[int] = set()

        for m in matches:
            if m.chunk_id in seen_chunk_ids:
                continue

            # Find neighbors: chunk_index - 1 to chunk_index + 1
            min_idx = max(0, m.chunk_index - max_window)
            max_idx = m.chunk_index + max_window

            neighbors = DocumentChunk.objects.filter(
                document_id=m.document_id,
                chunk_index__gte=min_idx,
                chunk_index__lte=max_idx,
            ).select_related("document").order_by("chunk_index")

            for c in neighbors:
                if c.id not in seen_chunk_ids:
                    seen_chunk_ids.add(c.id)
                    score = m.similarity_score if c.id == m.chunk_id else round(m.similarity_score * 0.95, 3)
                    expanded_chunks.append(
                        SearchResult(
                            chunk_id=c.id,
                            document_id=c.document_id,
                            document_title=c.document.title,
                            page_number=c.page_number,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            similarity_score=score,
                        )
                    )

        return expanded_chunks

    def search(self, query_embedding: List[float], top_k: int = 5,
               document_id: Optional[int] = None, min_similarity: float = 0.0,
               embedding_model: Optional[str] = None,
               page_number: Optional[int] = None) -> List[SearchResult]:
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
        if page_number is not None:
            qs = qs.filter(page_number=page_number)

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
