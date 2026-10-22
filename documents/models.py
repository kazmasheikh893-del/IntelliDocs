import os
import uuid

from django.db import models
from django.utils.text import get_valid_filename


def document_upload_path(instance, filename):
    """Collision-free, sanitized storage path (basename only, no traversal)."""
    safe_name = get_valid_filename(os.path.basename(filename)) or "document"
    return os.path.join("uploads", f"{uuid.uuid4().hex[:12]}_{safe_name}")


class Document(models.Model):
    """An uploaded document (PDF or TXT) ingested into IntelliDocs."""

    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_READY = "ready"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_READY, "Ready"),
        (STATUS_FAILED, "Failed"),
    ]

    title = models.CharField(max_length=255, help_text="Original document filename")
    file = models.FileField(upload_to=document_upload_path)
    file_type = models.CharField(max_length=10, help_text="pdf or txt")
    file_size = models.PositiveIntegerField(help_text="File size in bytes", default=0)
    content_hash = models.CharField(max_length=64, blank=True, default="", db_index=True,
                                    help_text="SHA-256 of the uploaded bytes (duplicate detection)")
    page_count = models.PositiveIntegerField(default=1, help_text="Number of pages detected")
    total_chunks = models.PositiveIntegerField(default=0, help_text="Number of indexed chunks")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"


class DocumentChunk(models.Model):
    """A text chunk with location metadata and the embedding that indexes it."""

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField(help_text="Sequential index of chunk in document")
    page_number = models.PositiveIntegerField(
        null=True, blank=True, help_text="1-based page number for PDFs, null for TXT"
    )
    content = models.TextField(help_text="Text content of the chunk")
    embedding = models.JSONField(default=list, blank=True, help_text="Vector embedding (list of floats)")
    embedding_model = models.CharField(max_length=100, blank=True, default="",
                                       help_text="Model that produced the embedding")
    embedding_dim = models.PositiveIntegerField(default=0, help_text="Embedding dimensionality")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document", "chunk_index"]
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
            models.Index(fields=["document", "page_number"]),
        ]

    def __str__(self):
        page_str = f"Page {self.page_number}" if self.page_number else "No Page"
        return f"[{self.document.title}] Chunk #{self.chunk_index} ({page_str})"
