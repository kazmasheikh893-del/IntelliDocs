from unittest.mock import patch, MagicMock
from django.test import TestCase, override_settings
from documents.models import Document, DocumentChunk
from documents.services.generator import (
    GeminiQAService,
    INSUFFICIENT_EVIDENCE_ANSWER,
)
from documents.services.vector_store import SearchResult


class GeminiQAServiceTests(TestCase):
    def setUp(self):
        self.doc = Document.objects.create(
            title="SampleDoc.pdf",
            file_type="pdf",
            status=Document.STATUS_READY,
        )

    @override_settings(GEMINI_API_KEY="test_gemini_api_key")
    @patch("documents.services.generator.genai.Client")
    @patch("documents.services.generator.GeminiEmbeddingService")
    def test_insufficient_evidence_when_no_vector_matches(self, mock_emb_service_cls, mock_client_cls):
        mock_emb = MagicMock()
        mock_emb.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_emb_service_cls.return_value = mock_emb

        mock_store = MagicMock()
        # Return empty list of matches (nothing meets similarity threshold)
        mock_store.search.return_value = []

        qa_service = GeminiQAService(
            api_key="test_key",
            vector_store=mock_store,
            similarity_threshold=0.5,
        )

        result = qa_service.answer_question("What is quantum computing?")
        self.assertEqual(result.answer, INSUFFICIENT_EVIDENCE_ANSWER)
        self.assertFalse(result.grounded)
        self.assertEqual(len(result.sources), 0)

    @override_settings(GEMINI_API_KEY="test_gemini_api_key")
    @patch("documents.services.generator.genai.Client")
    @patch("documents.services.generator.GeminiEmbeddingService")
    def test_grounded_answer_generation_with_citations(self, mock_emb_service_cls, mock_client_cls):
        mock_emb = MagicMock()
        mock_emb.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_emb_service_cls.return_value = mock_emb

        # Mock vector match
        mock_store = MagicMock()
        mock_store.search.return_value = [
            SearchResult(
                chunk_id=1,
                document_id=self.doc.id,
                document_title="SampleDoc.pdf",
                page_number=3,
                chunk_index=0,
                content="Deep learning models use multiple layers to progressively extract features.",
                similarity_score=0.88,
            )
        ]

        # Mock Gemini response
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_gen_response = MagicMock()
        mock_gen_response.text = "Deep learning uses multiple layers to extract features [Doc: SampleDoc.pdf, Page 3]."
        mock_client.models.generate_content.return_value = mock_gen_response

        qa_service = GeminiQAService(
            api_key="test_key",
            vector_store=mock_store,
        )

        result = qa_service.answer_question("How do deep learning models extract features?")
        self.assertTrue(result.grounded)
        self.assertIn("multiple layers", result.answer)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["page_number"], 3)
        self.assertEqual(result.sources[0]["document_title"], "SampleDoc.pdf")

    @override_settings(GEMINI_API_KEY="test_gemini_api_key")
    @patch("documents.services.generator.genai.Client")
    @patch("documents.services.generator.GeminiEmbeddingService")
    def test_empty_question_handling(self, mock_emb_service_cls, mock_client_cls):
        qa_service = GeminiQAService(api_key="test_key")
        result = qa_service.answer_question("   ")
        self.assertFalse(result.grounded)
        self.assertIn("valid question", result.answer)
