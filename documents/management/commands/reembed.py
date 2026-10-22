"""
Rebuild embeddings for stored chunks using the currently configured embedding model.

Use after changing GEMINI_EMBEDDING_MODEL / GEMINI_EMBEDDING_DIM. Chunk text is kept;
only vectors are replaced. Each document is updated in its own transaction, so a failure
leaves that document's previous vectors untouched. Requires a configured API key.

    python manage.py reembed
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from documents.models import Document
from documents.services.embeddings import GeminiEmbeddingService
from documents.services.errors import GeminiError


class Command(BaseCommand):
    help = "Re-embed all ready documents with the configured embedding model."

    def handle(self, *args, **options):
        try:
            service = GeminiEmbeddingService()
        except GeminiError as exc:
            raise CommandError(exc.user_message)

        done = 0
        for doc in Document.objects.filter(status=Document.STATUS_READY):
            chunks = list(doc.chunks.order_by("chunk_index"))
            if not chunks:
                continue
            if all(c.embedding_model == service.model and c.embedding_dim == service.dimensions
                   for c in chunks):
                continue
            try:
                vectors = service.embed_batch([c.content for c in chunks])
            except GeminiError as exc:
                raise CommandError(f"Stopped at '{doc.title}': {exc.user_message}")
            with transaction.atomic():
                for c, v in zip(chunks, vectors):
                    c.embedding, c.embedding_model, c.embedding_dim = v, service.model, len(v)
                    c.save(update_fields=["embedding", "embedding_model", "embedding_dim"])
            done += 1
            self.stdout.write(f"Re-embedded '{doc.title}' ({len(chunks)} chunks)")
        self.stdout.write(self.style.SUCCESS(f"Done. {done} document(s) re-embedded."))
