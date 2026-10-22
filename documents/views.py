import json
import logging
import os
from functools import wraps

from django.conf import settings
from django.contrib import messages
from django.db.models import Sum
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from .forms import DocumentUploadForm
from .models import Document
from .services.embeddings import PLACEHOLDER_KEYS
from .services.errors import GeminiError
from .services.generator import GeminiQAService, INSUFFICIENT_EVIDENCE_ANSWER
from .services.ingestion import IngestionService

logger = logging.getLogger(__name__)


def auth_required(view_func):
    """Enforces authentication only when settings.REQUIRE_AUTH is enabled."""
    @wraps(view_func)
    def _wrapped(request: HttpRequest, *args, **kwargs):
        if getattr(settings, 'REQUIRE_AUTH', False) and not request.user.is_authenticated:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest' or (request.content_type or '').startswith('application/json'):
                return JsonResponse({'error': 'Authentication required. Please log in.'}, status=401)
            return redirect(f"{reverse('login')}?next={request.path}")
        return view_func(request, *args, **kwargs)
    return _wrapped


def _gemini_configured() -> bool:
    return (getattr(settings, "GEMINI_API_KEY", "") or "").strip() not in PLACEHOLDER_KEYS


@ensure_csrf_cookie
@auth_required
def index_view(request: HttpRequest) -> HttpResponse:
    """Dashboard: statistics are computed from persisted records."""
    documents = Document.objects.all()
    ready = documents.filter(status=Document.STATUS_READY)
    return render(request, "documents/index.html", {
        "documents": documents,
        "upload_form": DocumentUploadForm(),
        "total_docs": documents.count(),
        "ready_docs": ready.count(),
        "failed_docs": documents.filter(status=Document.STATUS_FAILED).count(),
        "total_chunks": ready.aggregate(n=Sum("total_chunks"))["n"] or 0,
        "gemini_configured": _gemini_configured(),
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
    })


@require_POST
@auth_required
def upload_document_view(request: HttpRequest) -> HttpResponse:
    form = DocumentUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        for errors in form.errors.values():
            for error in errors:
                messages.error(request, f"Upload error: {error}")
        return redirect("documents:index")

    uploaded = form.cleaned_data["file"]
    doc = form.save(commit=False)
    doc.title = os.path.basename(uploaded.name)[:255]
    doc.file_type = os.path.splitext(doc.title)[1].lower().lstrip(".")
    doc.file_size = uploaded.size
    doc.content_hash = getattr(form, "content_hash", "")
    doc.status = Document.STATUS_PENDING
    doc.save()

    IngestionService.process_document(doc)
    if doc.status == Document.STATUS_READY:
        messages.success(
            request, f"'{doc.title}' processed: {doc.total_chunks} passages indexed."
        )
    else:
        messages.error(request, f"Processing failed for '{doc.title}': {doc.error_message}")
    return redirect("documents:index")


@ensure_csrf_cookie
@auth_required
def qa_view(request: HttpRequest) -> HttpResponse:
    ready_documents = Document.objects.filter(status=Document.STATUS_READY)
    return render(request, "documents/chat.html", {
        "documents": ready_documents,
        "gemini_configured": _gemini_configured(),
        "generation_model": settings.GEMINI_GENERATION_MODEL,
        "fallback_text": INSUFFICIENT_EVIDENCE_ANSWER,
        "selected_doc": request.GET.get("doc", ""),
    })


@require_POST
@auth_required
def query_api_view(request: HttpRequest) -> JsonResponse:
    """POST {"question": str, "document_id": int|null} -> {answer, grounded, sources}."""
    try:
        data = json.loads(request.body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    question = str(data.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Question cannot be empty."}, status=400)

    doc_id = data.get("document_id")
    if doc_id in (None, ""):
        doc_id = None
    else:
        try:
            doc_id = int(doc_id)
        except (TypeError, ValueError):
            return JsonResponse({"error": "Invalid document selection."}, status=400)
        if not Document.objects.filter(pk=doc_id, status=Document.STATUS_READY).exists():
            return JsonResponse({"error": "The selected document is not available."}, status=404)

    try:
        result = GeminiQAService().answer_question(question=question, document_id=doc_id)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except GeminiError as exc:
        return JsonResponse({"error": exc.user_message}, status=exc.http_status)
    except Exception:
        logger.exception("Unexpected error while answering a question")
        return JsonResponse({"error": "An unexpected error occurred. Please try again."}, status=500)

    response_data = {
        "question": result.question,
        "answer": result.answer,
        "grounded": result.grounded,
        "sources": result.sources,
    }
    if settings.DEBUG and result.diagnostics:
        response_data["diagnostics"] = result.diagnostics

    return JsonResponse(response_data)


@require_POST
@auth_required
def delete_document_view(request: HttpRequest, document_id: int) -> HttpResponse:
    doc = get_object_or_404(Document, id=document_id)
    title = doc.title
    if doc.file:
        doc.file.delete(save=False)
    doc.delete()
    messages.success(request, f"Document '{title}' was deleted.")
    return redirect("documents:index")
