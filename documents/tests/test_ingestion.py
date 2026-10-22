import shutil
import tempfile

from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from documents.models import Document, DocumentChunk
from documents.services.errors import QuotaExceededError
from documents.services.ingestion import IngestionService
from .helpers import FakeEmbeddingService, make_blank_pdf, make_pdf

TMP_MEDIA = tempfile.mkdtemp(prefix="intellidocs-test-")


def tearDownModule():
    shutil.rmtree(TMP_MEDIA, ignore_errors=True)


@override_settings(MEDIA_ROOT=TMP_MEDIA, RAG_CHUNK_SIZE=200, RAG_CHUNK_OVERLAP=30)
class IngestionTests(TestCase):
    def _doc(self, name, data):
        ftype = name.rsplit(".", 1)[1]
        doc = Document(title=name, file_type=ftype, file_size=len(data))
        doc.file.save(name, ContentFile(data), save=False)
        doc.save()
        return doc

    def test_pdf_ingestion_persists_chunks_with_metadata(self):
        doc = self._doc("facts.pdf", make_pdf(["Mount Everest is 8849 metres tall.", "The Nile flows north."]))
        svc = FakeEmbeddingService(model="test-embed", dimensions=8)
        IngestionService.process_document(doc, embedding_service=svc)
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.page_count, 2)
        self.assertEqual(doc.total_chunks, doc.chunks.count())
        self.assertEqual(sorted({c.page_number for c in doc.chunks.all()}), [1, 2])
        for c in doc.chunks.all():
            self.assertEqual(c.embedding_model, "test-embed")
            self.assertEqual(c.embedding_dim, 8)
            self.assertEqual(len(c.embedding), 8)
        page2 = doc.chunks.get(page_number=2)
        self.assertIn("Nile", page2.content)

    def test_txt_ingestion(self):
        doc = self._doc("notes.txt", b"Plain notes about retrieval.")
        IngestionService.process_document(doc, embedding_service=FakeEmbeddingService())
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_READY)
        self.assertEqual(doc.page_count, 1)
        self.assertIsNone(doc.chunks.first().page_number)

    def test_scanned_pdf_fails_visibly_without_chunks(self):
        doc = self._doc("scan.pdf", make_blank_pdf(2))
        IngestionService.process_document(doc, embedding_service=FakeEmbeddingService())
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_FAILED)
        self.assertIn("readable text", doc.error_message)
        self.assertEqual(DocumentChunk.objects.filter(document=doc).count(), 0)
        self.assertEqual(doc.total_chunks, 0)

    def test_corrupt_pdf_fails(self):
        doc = self._doc("bad.pdf", b"%PDF-1.4 garbage")
        IngestionService.process_document(doc, embedding_service=FakeEmbeddingService())
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_FAILED)

    def test_embedding_failure_leaves_no_partial_or_ready_state(self):
        doc = self._doc("ok.txt", b"Some perfectly fine text. " * 40)
        svc = FakeEmbeddingService()
        svc.fail_with = QuotaExceededError()
        IngestionService.process_document(doc, embedding_service=svc)
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_FAILED)
        self.assertEqual(doc.error_message, QuotaExceededError.user_message)
        self.assertEqual(doc.chunks.count(), 0)
        self.assertFalse(doc.file)  # stored upload removed on failure

    def test_missing_api_key_marks_failed_with_clear_message(self):
        doc = self._doc("ok.txt", b"Text that needs an embedding.")
        with override_settings(GEMINI_API_KEY=""):
            IngestionService.process_document(doc)  # real service -> no key
        doc.refresh_from_db()
        self.assertEqual(doc.status, Document.STATUS_FAILED)
        self.assertIn("GEMINI_API_KEY", doc.error_message)

    def test_reprocessing_replaces_chunks_without_duplicates(self):
        doc = self._doc("r.txt", b"Reprocess me please. " * 30)
        svc = FakeEmbeddingService()
        IngestionService.process_document(doc, embedding_service=svc)
        first = doc.chunks.count()
        IngestionService.process_document(doc, embedding_service=svc)
        self.assertEqual(doc.chunks.count(), first)
