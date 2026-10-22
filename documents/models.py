import os
import uuid
from django.db import models


def document_upload_path(instance, filename):
    """Generate a clean, collision-free storage path for uploaded files."""
    ext = os.path.splitext(filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:12]}_{filename}"
    return os.path.join("uploads", unique_name)


class Document(models.Model):
    """Represents an uploaded document (PDF or TXT) ingested into IntelliDocs."""

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
    page_count = models.PositiveIntegerField(default=1, help_text="Number of pages detected")
    total_chunks = models.PositiveIntegerField(default=0, help_text="Number of extracted chunks")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"


class DocumentChunk(models.Model):
    """Represents a text chunk extracted from a document with location metadata and embeddings."""

    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    chunk_index = models.PositiveIntegerField(help_text="Sequential index of chunk in document")
    page_number = models.PositiveIntegerField(
        null=True, blank=True, help_text="1-based page number for PDFs, null for TXT"
    )
    content = models.TextField(help_text="Text content of the chunk")
    embedding = models.JSONField(
        default=list, blank=True, help_text="Vector embedding as list of floats"
    )
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
