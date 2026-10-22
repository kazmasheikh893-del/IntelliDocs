"""
Vector storage and similarity retrieval service.
Provides an abstract BaseVectorStore and a zero-dependency Django ORM + NumPy implementation.
"""

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
    """Abstract interface for vector storage and retrieval."""

    @abstractmethod
    def search(
        self,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        document_id: Optional[int] = None,
        min_similarity: float = 0.0,
        query_vector: Optional[List[float]] = None,
    ) -> List[SearchResult]:
        pass


class ORMVectorStore(BaseVectorStore):
    """
    Stores embeddings in Django ORM JSONField and computes cosine similarity
    in-memory using vector matrix operations in NumPy.
    """

    def search(
        self,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        document_id: Optional[int] = None,
        min_similarity: float = 0.0,
        query_vector: Optional[List[float]] = None,
    ) -> List[SearchResult]:
        target_vec = query_embedding or query_vector
        if not target_vec:
            return []

        # Filter chunks that have embeddings from READY documents
        qs = DocumentChunk.objects.select_related("document").filter(
            document__status=Document.STATUS_READY
        ).exclude(embedding=[])

        if document_id:
            qs = qs.filter(document_id=document_id)

        chunks = list(qs)
        if not chunks:
            return []

        # Convert embeddings into a 2D numpy matrix
        embeddings_matrix = np.array([c.embedding for c in chunks], dtype=np.float32)
        q_vector = np.array(target_vec, dtype=np.float32)

        # Normalize query vector
        q_norm = np.linalg.norm(q_vector)
        if q_norm == 0:
            return []
        q_vector_normalized = q_vector / q_norm

        # Normalize matrix rows
        row_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        row_norms[row_norms == 0] = 1.0
        normalized_matrix = embeddings_matrix / row_norms

        # Dot product yields cosine similarity scores
        scores = np.dot(normalized_matrix, q_vector_normalized)

        # Pair chunks with scores
        scored_results = []
        for idx, score in enumerate(scores):
            score_val = float(score)
            if score_val >= min_similarity:
                chunk = chunks[idx]
                scored_results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document.id,
                        document_title=chunk.document.title,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        similarity_score=score_val,
                    )
                )

        # Sort descending by score and take top_k
        scored_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored_results[:top_k]
