import json
from django.test import TestCase, Client
from django.urls import reverse
from documents.models import Document, DocumentChunk


class DocumentViewsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.doc = Document.objects.create(
            title="SampleDocument.pdf",
            file_type="pdf",
            status=Document.STATUS_READY,
            page_count=2,
            total_chunks=3,
        )

    def test_index_view_loads_successfully(self):
        response = self.client.get(reverse("documents:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SampleDocument.pdf")
        self.assertContains(response, "Document Repository")

    def test_qa_view_loads_successfully(self):
        response = self.client.get(reverse("documents:qa"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Document Filter")
        self.assertContains(response, "Ask Gemini")

    def test_query_api_rejects_empty_question(self):
        url = reverse("documents:api_query")
        response = self.client.post(
            url,
            data=json.dumps({"question": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_delete_document_view(self):
        doc_to_delete = Document.objects.create(
            title="ToDelete.txt",
            file_type="txt",
            status=Document.STATUS_READY,
        )
        DocumentChunk.objects.create(
            document=doc_to_delete,
            chunk_index=0,
            content="Some text",
        )

        url = reverse("documents:delete", kwargs={"document_id": doc_to_delete.id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Document.objects.filter(id=doc_to_delete.id).exists())
        self.assertFalse(DocumentChunk.objects.filter(document_id=doc_to_delete.id).exists())
