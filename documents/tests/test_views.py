import json
import shutil
import tempfile
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from documents.models import Document, DocumentChunk
from documents.services.errors import QuotaExceededError
from documents.services.generator import QAResult
from .helpers import FakeEmbeddingService, make_blank_pdf, make_pdf

TMP_MEDIA = tempfile.mkdtemp(prefix="intellidocs-views-")


def tearDownModule():
    shutil.rmtree(TMP_MEDIA, ignore_errors=True)


def post_json(client, url, payload):
    return client.post(url, data=json.dumps(payload), content_type="application/json")


class DashboardTests(TestCase):
    def test_empty_dashboard(self):
        r = self.client.get(reverse("documents:index"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "No documents uploaded yet")
        self.assertEqual(r.context["total_docs"], 0)
        self.assertEqual(r.context["total_chunks"], 0)

    def test_dashboard_reflects_persisted_records(self):
        ready = Document.objects.create(title="a.pdf", file_type="pdf", status=Document.STATUS_READY, total_chunks=7)
        Document.objects.create(title="b.txt", file_type="txt", status=Document.STATUS_READY, total_chunks=3)
        Document.objects.create(title="c.pdf", file_type="pdf", status=Document.STATUS_FAILED,
                                error_message="No readable text could be extracted")
        r = self.client.get(reverse("documents:index"))
        self.assertEqual((r.context["total_docs"], r.context["ready_docs"],
                          r.context["failed_docs"], r.context["total_chunks"]), (3, 2, 1, 10))
        self.assertContains(r, "a.pdf")
        self.assertContains(r, "No readable text could be extracted")
        self.assertContains(r, "Failed")

    def test_qa_page_lists_only_ready_documents(self):
        Document.objects.create(title="ready.pdf", file_type="pdf", status=Document.STATUS_READY)
        Document.objects.create(title="broken.pdf", file_type="pdf", status=Document.STATUS_FAILED)
        r = self.client.get(reverse("documents:qa"))
        self.assertContains(r, "ready.pdf")
        self.assertNotContains(r, "broken.pdf")
        self.assertContains(r, reverse("documents:api_query"))  # frontend URL matches routing

    def test_qa_page_without_documents_shows_empty_state(self):
        r = self.client.get(reverse("documents:qa"))
        self.assertContains(r, "No documents are ready yet")

    @override_settings(GEMINI_API_KEY="")
    def test_missing_key_warning_shown(self):
        self.assertContains(self.client.get(reverse("documents:index")), "GEMINI_API_KEY is not configured")

    @override_settings(GEMINI_API_KEY="super-secret-value-xyz")
    def test_api_key_never_rendered(self):
        for name in ("documents:index", "documents:qa"):
            self.assertNotContains(self.client.get(reverse(name)), "super-secret-value-xyz")


class QueryApiTests(TestCase):
    url = reverse("documents:api_query")

    def setUp(self):
        self.doc = Document.objects.create(title="d.pdf", file_type="pdf", status=Document.STATUS_READY)

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_csrf_enforced(self):
        c = Client(enforce_csrf_checks=True)
        self.assertEqual(post_json(c, self.url, {"question": "hi"}).status_code, 403)

    def test_invalid_payloads(self):
        self.assertEqual(self.client.post(self.url, data="{bad", content_type="application/json").status_code, 400)
        self.assertEqual(post_json(self.client, self.url, ["x"]).status_code, 400)
        self.assertEqual(post_json(self.client, self.url, {"question": "  "}).status_code, 400)
        self.assertEqual(post_json(self.client, self.url, {"question": "q", "document_id": "abc"}).status_code, 400)
        self.assertEqual(post_json(self.client, self.url, {"question": "q", "document_id": 9999}).status_code, 404)

    def test_failed_document_cannot_be_queried(self):
        bad = Document.objects.create(title="x", file_type="pdf", status=Document.STATUS_FAILED)
        self.assertEqual(post_json(self.client, self.url, {"question": "q", "document_id": bad.id}).status_code, 404)

    @override_settings(GEMINI_API_KEY="")
    def test_missing_key_gives_clear_503(self):
        r = post_json(self.client, self.url, {"question": "q"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("GEMINI_API_KEY", r.json()["error"])

    @override_settings(GEMINI_API_KEY="k")
    @patch("documents.views.GeminiQAService")
    def test_success_response_structure(self, svc):
        svc.return_value.answer_question.return_value = QAResult(
            question="q", answer="A [d.pdf, p. 2]", grounded=True,
            sources=[{"snippet_index": 1, "document_id": self.doc.id, "document_title": "d.pdf",
                      "page_number": 2, "similarity_score": 0.8, "excerpt": "..."}])
        r = post_json(self.client, self.url, {"question": "q", "document_id": self.doc.id})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(set(data), {"question", "answer", "grounded", "sources"})
        self.assertEqual(data["sources"][0]["page_number"], 2)
        svc.return_value.answer_question.assert_called_once_with(question="q", document_id=self.doc.id)

    @override_settings(GEMINI_API_KEY="k")
    @patch("documents.views.GeminiQAService")
    def test_gemini_errors_are_mapped_to_safe_messages(self, svc):
        svc.return_value.answer_question.side_effect = QuotaExceededError()
        r = post_json(self.client, self.url, {"question": "q"})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"], QuotaExceededError.user_message)

    @override_settings(GEMINI_API_KEY="k")
    @patch("documents.views.GeminiQAService")
    def test_unexpected_errors_do_not_leak_details(self, svc):
        svc.return_value.answer_question.side_effect = RuntimeError("secret internals /etc/passwd k")
        r = post_json(self.client, self.url, {"question": "q"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("internals", r.content.decode())
        self.assertNotIn("Traceback", r.content.decode())


@override_settings(MEDIA_ROOT=TMP_MEDIA, GEMINI_API_KEY="k")
class UploadViewTests(TestCase):
    url = reverse("documents:upload")

    def _upload(self, name, data, client=None):
        return (client or self.client).post(self.url, {"file": SimpleUploadedFile(name, data)}, follow=True)

    def test_csrf_enforced(self):
        c = Client(enforce_csrf_checks=True)
        r = c.post(self.url, {"file": SimpleUploadedFile("a.txt", b"hello")})
        self.assertEqual(r.status_code, 403)

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    @patch("documents.services.ingestion.GeminiEmbeddingService")
    def test_successful_pdf_upload_end_to_end_persistence(self, emb):
        emb.return_value = FakeEmbeddingService()
        r = self._upload("facts.pdf", make_pdf(["Fact one is here.", "Fact two is there."]))
        self.assertContains(r, "processed")
        doc = Document.objects.get(title="facts.pdf")
        self.assertEqual((doc.status, doc.page_count), (Document.STATUS_READY, 2))
        self.assertEqual(doc.total_chunks, DocumentChunk.objects.filter(document=doc).count())
        self.assertEqual(len(doc.content_hash), 64)

    def test_missing_key_results_in_visible_failed_document(self):
        with override_settings(GEMINI_API_KEY=""):
            r = self._upload("n.txt", b"Some content to embed.")
        doc = Document.objects.get(title="n.txt")
        self.assertEqual(doc.status, Document.STATUS_FAILED)
        self.assertContains(r, "GEMINI_API_KEY")
        self.assertEqual(doc.chunks.count(), 0)

    def test_scanned_pdf_reports_failure(self):
        with patch("documents.services.ingestion.GeminiEmbeddingService") as emb:
            emb.return_value = FakeEmbeddingService()
            r = self._upload("scan.pdf", make_blank_pdf())
        self.assertContains(r, "readable text")
        self.assertEqual(Document.objects.get(title="scan.pdf").status, Document.STATUS_FAILED)

    def test_rejected_uploads_create_no_records(self):
        self.assertContains(self._upload("evil.exe", b"MZ"), "Unsupported file format")
        self.assertContains(self._upload("empty.txt", b""), "empty")
        self.assertContains(self._upload("fake.pdf", b"not a pdf"), "not a valid PDF")
        self.assertContains(self._upload("bin.txt", b"abc\x00def"), "binary")
        self.assertEqual(Document.objects.count(), 0)

    @override_settings(MAX_UPLOAD_SIZE_BYTES=100)
    def test_oversized_upload_rejected(self):
        self.assertContains(self._upload("big.txt", b"x" * 500), "exceeds")
        self.assertEqual(Document.objects.count(), 0)

    @patch("documents.services.ingestion.GeminiEmbeddingService")
    def test_duplicate_upload_rejected(self, emb):
        emb.return_value = FakeEmbeddingService()
        self._upload("one.txt", b"identical content here")
        r = self._upload("two.txt", b"identical content here")
        self.assertContains(r, "already been uploaded")
        self.assertEqual(Document.objects.count(), 1)

    @patch("documents.services.ingestion.GeminiEmbeddingService")
    def test_path_traversal_filename_is_neutralised(self, emb):
        emb.return_value = FakeEmbeddingService()
        self._upload("../../evil.txt", b"harmless text")
        doc = Document.objects.get()
        self.assertEqual(doc.title, "evil.txt")
        self.assertTrue(doc.file.path.startswith(TMP_MEDIA))


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class DeleteViewTests(TestCase):
    def test_delete_removes_document_and_chunks(self):
        d = Document.objects.create(title="t.txt", file_type="txt", status=Document.STATUS_READY)
        DocumentChunk.objects.create(document=d, chunk_index=0, content="x")
        r = self.client.post(reverse("documents:delete", args=[d.id]))
        self.assertEqual(r.status_code, 302)
        self.assertFalse(Document.objects.exists())
        self.assertFalse(DocumentChunk.objects.exists())

    def test_delete_requires_post_and_csrf(self):
        d = Document.objects.create(title="t.txt", file_type="txt")
        self.assertEqual(self.client.get(reverse("documents:delete", args=[d.id])).status_code, 405)
        c = Client(enforce_csrf_checks=True)
        self.assertEqual(c.post(reverse("documents:delete", args=[d.id])).status_code, 403)
        self.assertTrue(Document.objects.exists())
