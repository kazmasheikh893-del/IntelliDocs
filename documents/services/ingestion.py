"""
Document ingestion pipeline orchestrating extraction, chunking,
embedding generation, and vector persistence.
"""

import logging
from django.db import transaction
from documents.models import Document, DocumentChunk
from .extractor import DocumentExtractor, ExtractionError
from .chunker import TextChunker
from .embeddings import GeminiEmbeddingService, MissingApiKeyError, EmbeddingError

logger = logging.getLogger(__name__)


class IngestionService:
    """Manages the lifecycle of extracting, chunking, and embedding uploaded documents."""

    @classmethod
    def process_document(cls, document: Document) -> Document:
        """
        Executes the full ingestion pipeline for an uploaded Document.
        Updates document status and logs any failure.
        """
        document.status = Document.STATUS_PROCESSING
        document.error_message = ""
        document.save(update_fields=["status", "error_message"])

        try:
            # 1. Extraction
            pages_data = DocumentExtractor.extract(document.file.path, document.title)
            
            # Detect page count
            if document.file_type == "pdf":
                document.page_count = len(pages_data)
            else:
                document.page_count = 1

            # 2. Chunking
            chunker = TextChunker(chunk_size=800, chunk_overlap=100)
            text_chunks = chunker.chunk_pages(pages_data)

            if not text_chunks:
                raise ExtractionError("Chunking resulted in 0 usable text passages.")

            # 3. Generate Embeddings
            embedding_service = GeminiEmbeddingService()
            texts = [c.content for c in text_chunks]
            embeddings = embedding_service.embed_batch(texts)

            if len(embeddings) != len(text_chunks):
                raise EmbeddingError("Mismatch between count of chunks and generated embeddings.")

            # 4. Save Chunks transactionally
            with transaction.atomic():
                # Clean up any existing chunks for this document
                document.chunks.all().delete()

                chunk_objects = [
                    DocumentChunk(
                        document=document,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        content=c.content,
                        embedding=embeddings[idx],
                    )
                    for idx, c in enumerate(text_chunks)
                ]
                DocumentChunk.objects.bulk_create(chunk_objects)

                document.total_chunks = len(chunk_objects)
                document.status = Document.STATUS_READY
                document.error_message = ""
                document.save()

            return document

        except (ExtractionError, MissingApiKeyError, EmbeddingError, Exception) as e:
            error_msg = str(e)
            logger.error(f"Failed to ingest document '{document.title}': {error_msg}")
            document.status = Document.STATUS_FAILED
            document.error_message = error_msg
            document.save(update_fields=["status", "error_message"])
            raise
