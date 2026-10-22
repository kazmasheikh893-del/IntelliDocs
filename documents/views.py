import json
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.contrib import messages
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import Document
from .forms import DocumentUploadForm
from .services.ingestion import IngestionService
from .services.generator import GeminiQAService, INSUFFICIENT_EVIDENCE_ANSWER
from .services.embeddings import MissingApiKeyError


def index_view(request: HttpRequest) -> HttpResponse:
    """Dashboard displaying uploaded documents and the document upload form."""
    documents = Document.objects.all()
    upload_form = DocumentUploadForm()
    
    # Calculate statistics
    total_docs = documents.count()
    ready_docs = documents.filter(status=Document.STATUS_READY).count()
    total_chunks = sum(d.total_chunks for d in documents)

    return render(
        request,
        "documents/index.html",
        {
            "documents": documents,
            "upload_form": upload_form,
            "total_docs": total_docs,
            "ready_docs": ready_docs,
            "total_chunks": total_chunks,
        },
    )


@require_POST
def upload_document_view(request: HttpRequest) -> HttpResponse:
    """Handles document upload and triggers ingestion."""
    form = DocumentUploadForm(request.POST, request.FILES)
    if form.is_valid():
        uploaded_file = form.cleaned_data["file"]
        ext = os.path.splitext(uploaded_file.name)[1].lower().replace(".", "")
        
        doc = form.save(commit=False)
        doc.title = uploaded_file.name
        doc.file_type = ext
        doc.file_size = uploaded_file.size
        doc.status = Document.STATUS_PENDING
        doc.save()

        # Trigger ingestion
        try:
            IngestionService.process_document(doc)
            messages.success(
                request,
                f"Document '{doc.title}' successfully processed ({doc.total_chunks} chunks indexed)."
            )
        except MissingApiKeyError as e:
            messages.warning(
                request,
                f"Document saved, but embedding failed: {str(e)}. Please configure GEMINI_API_KEY in .env."
            )
        except Exception as e:
            messages.error(request, f"Processing failed for '{doc.title}': {str(e)}")

        return redirect("documents:index")
    else:
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(request, f"Upload error: {error}")
        return redirect("documents:index")


@ensure_csrf_cookie
def qa_view(request: HttpRequest) -> HttpResponse:
    """Interactive Document Intelligence Q&A Interface."""
    ready_documents = Document.objects.filter(status=Document.STATUS_READY)
    return render(
        request,
        "documents/chat.html",
        {
            "documents": ready_documents,
            "default_insufficient_answer": INSUFFICIENT_EVIDENCE_ANSWER,
        },
    )


@require_POST
def query_api_view(request: HttpRequest) -> JsonResponse:
    """
    JSON API endpoint for answering questions against ingested documents.
    Payload: { "question": str, "document_id": int | null }
    """
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    question = (data.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Question cannot be empty."}, status=400)

    doc_id = data.get("document_id")
    if doc_id:
        try:
            doc_id = int(doc_id)
        except ValueError:
            doc_id = None

    try:
        qa_service = GeminiQAService()
        result = qa_service.answer_question(question=question, document_id=doc_id)
        return JsonResponse({
            "question": result.question,
            "answer": result.answer,
            "grounded": result.grounded,
            "sources": result.sources,
        })
    except MissingApiKeyError as e:
        return JsonResponse({
            "error": "Gemini API key is not configured.",
            "details": str(e),
        }, status=503)
    except Exception as e:
        return JsonResponse({
            "error": "Failed to generate answer.",
            "details": str(e),
        }, status=500)


@require_POST
def delete_document_view(request: HttpRequest, document_id: int) -> HttpResponse:
    """Deletes a document, its database chunks, and local uploaded file."""
    doc = get_object_or_404(Document, id=document_id)
    doc_title = doc.title
    
    # Delete file from disk if present
    if doc.file and os.path.exists(doc.file.path):
        try:
            os.remove(doc.file.path)
        except Exception:
            pass

    doc.delete()
    messages.success(request, f"Document '{doc_title}' was successfully deleted.")
    return redirect("documents:index")
