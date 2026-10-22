"""
Document ingestion pipeline: extract -> chunk -> embed -> persist.

All fallible work (extraction, chunking, embedding API calls) happens BEFORE anything is
written to the chunk table, and chunk creation + the 'ready' status flip happen in a single
transaction. A document is therefore never marked ready without a complete set of chunks,
and a failed run never leaves partial chunks behind.
"""

import logging

from django.conf import settings
from django.db import transaction

from documents.models import Document, DocumentChunk
from .chunker import TextChunker
from .embeddings import GeminiEmbeddingService
from .errors import GeminiError
from .extractor import DocumentExtractor, ExtractionError

logger = logging.getLogger(__name__)


class IngestionService:
    @classmethod
    def process_document(cls, document: Document, embedding_service=None) -> Document:
        """Run the pipeline. Never raises for expected failures; inspect `document.status`."""
        document.status = Document.STATUS_PROCESSING
        document.error_message = ""
        document.save(update_fields=["status", "error_message", "updated_at"])

        try:
            pages = DocumentExtractor.extract(document.file.path, f"file.{document.file_type}")
            page_count = len(pages) if document.file_type == "pdf" else 1

            chunker = TextChunker(settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP)
            chunks = chunker.chunk_pages(pages)
            if not chunks:
                raise ExtractionError("No usable text passages were found in this document.")

            service = embedding_service or GeminiEmbeddingService()
            vectors = service.embed_batch([c.content for c in chunks])
            if len(vectors) != len(chunks):
                raise GeminiError("Embedding count did not match chunk count; nothing was saved.")

            with transaction.atomic():
                document.chunks.all().delete()
                DocumentChunk.objects.bulk_create([
                    DocumentChunk(
                        document=document,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        content=c.content,
                        embedding=vectors[i],
                        embedding_model=service.model,
                        embedding_dim=len(vectors[i]),
                    )
                    for i, c in enumerate(chunks)
                ])
                document.page_count = page_count
                document.total_chunks = len(chunks)
                document.status = Document.STATUS_READY
                document.error_message = ""
                document.save()
            return document

        except (ExtractionError, GeminiError) as exc:
            message = getattr(exc, "user_message", None) or str(exc)
            cls._mark_failed(document, message)
        except Exception:
            logger.exception("Unexpected ingestion failure for document id=%s", document.pk)
            cls._mark_failed(document, "An unexpected error occurred while processing this document.")
        return document

    @staticmethod
    def _mark_failed(document: Document, message: str) -> None:
        logger.info("Ingestion failed for document id=%s: %s", document.pk, message)
        document.chunks.all().delete()
        document.total_chunks = 0
        document.status = Document.STATUS_FAILED
        document.error_message = message[:1000]
        document.save()
        # Remove the stored upload; the failed record stays visible on the dashboard.
        try:
            if document.file:
                document.file.delete(save=True)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove stored file for document id=%s", document.pk)
