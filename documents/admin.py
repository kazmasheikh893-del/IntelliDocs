from django.contrib import admin
from .models import Document, DocumentChunk


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ("chunk_index", "page_number", "content_preview", "created_at")
    fields = ("chunk_index", "page_number", "content_preview", "created_at")
    can_delete = False

    def content_preview(self, obj):
        return obj.content[:100] + ("..." if len(obj.content) > 100 else "")
    content_preview.short_description = "Content Preview"


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "file_type", "page_count", "total_chunks", "status", "created_at")
    list_filter = ("status", "file_type", "created_at")
    search_fields = ("title",)
    readonly_fields = ("created_at", "updated_at", "file_size", "page_count", "total_chunks")
    inlines = [DocumentChunkInline]


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "chunk_index", "page_number", "content_preview", "created_at")
    list_filter = ("document", "page_number")
    search_fields = ("content", "document__title")
    readonly_fields = ("document", "chunk_index", "page_number", "content", "embedding", "created_at")

    def content_preview(self, obj):
        return obj.content[:100] + ("..." if len(obj.content) > 100 else "")
    content_preview.short_description = "Snippet"
