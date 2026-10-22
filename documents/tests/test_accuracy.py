"""
Accuracy and regression test suite for IntelliDocs.
Covers:
- Query intent classification and page targeting
- Document-wide summarization across all pages
- Technical definition and plain-language explanation modes
- Intel 8051 8-bit microcontroller precision (OCR ligature defense)
- Multi-citation parsing and trusted metadata formatting
- Diagnostics payload in DEBUG mode
"""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from django.test import TestCase, override_settings

from documents.models import Document, DocumentChunk
from documents.services.generator import GeminiQAService, INSUFFICIENT_EVIDENCE_ANSWER
from documents.services.query_analyzer import QueryAnalyzer, QueryIntent
from documents.services.summarizer import DocumentSummarizer
from documents.services.vector_store import SearchResult
from .helpers import FakeEmbeddingService
from .test_vector_store import mk_chunk

E1 = [1.0, 0, 0, 0]
E2 = [0, 1.0, 0, 0]


def mock_genai_reply(text_dict):
    return SimpleNamespace(text=json.dumps(text_dict))


class QueryAnalyzerTests(TestCase):
    def test_document_summary_intents(self):
        queries = [
            "Analyze all 9 pages of this document and identify all topics",
            "Give me a complete summary of the entire document",
            "What is the document-level summary?",
            "Provide exam-relevant material and study roadmap for the whole document",
        ]
        for q in queries:
            res = QueryAnalyzer.analyze(q)
            self.assertEqual(res.intent, QueryIntent.DOCUMENT_SUMMARY, f"Failed for: {q}")

    def test_page_specific_intents(self):
        res = QueryAnalyzer.analyze("Summarize page 2 of the document")
        self.assertEqual(res.intent, QueryIntent.PAGE_SPECIFIC)
        self.assertEqual(res.target_page, 2)

        res2 = QueryAnalyzer.analyze("What is described on page 5?")
        self.assertEqual(res2.intent, QueryIntent.PAGE_SPECIFIC)
        self.assertEqual(res2.target_page, 5)

    def test_definition_intents_with_simple_words(self):
        res = QueryAnalyzer.analyze(
            "What is the definition of a microcomputer in this document? Also explain in simple words and provide the source."
        )
        self.assertEqual(res.intent, QueryIntent.DEFINITION)
        self.assertTrue(res.wants_simple_explanation)

    def test_comparison_intents(self):
        res = QueryAnalyzer.analyze("Compare microprocessor and microcontroller architecture")
        self.assertEqual(res.intent, QueryIntent.COMPARISON)


@override_settings(GEMINI_API_KEY="test-key", RAG_MIN_SIMILARITY=0.5, RAG_TOP_K=5, DEBUG=True)
class AccuracyAndRegressionTests(TestCase):
    def setUp(self):
        self.doc = Document.objects.create(
            title="GENERIC _UNIT 1.pdf",
            file_type="pdf",
            status=Document.STATUS_READY,
            page_count=9,
        )
        # Create chunks across multiple pages
        self.chunk_p1 = mk_chunk(
            self.doc, 0, 1,
            "The Intel 8051 (1980) became a very popular and influential l8-bit microcontroller.",
            E1, model="test-embed"
        )
        self.chunk_p2 = mk_chunk(
            self.doc, 1, 2,
            "A microcomputer is a small, relatively inexpensive computer with a microprocessor as its CPU. "
            "It includes memory (RAM/ROM) and input/output (I/O) facilities on a printed circuit board (PCB).",
            E1, model="test-embed"
        )
        self.chunk_p6 = mk_chunk(
            self.doc, 2, 6,
            "8051: An early and widely used 8-bit microcontroller family known for its simple architecture.",
            E1, model="test-embed"
        )
        self.emb = FakeEmbeddingService(
            model="test-embed",
            dimensions=4,
            vectors={"microcomputer": E1, "8051": E1, "summary": E1, "default": E1},
        )
        patcher = patch("documents.services.generator.genai.Client")
        self.mock_client_cls = patcher.start()
        self.mock_client = self.mock_client_cls.return_value
        self.addCleanup(patcher.stop)

    def svc(self):
        return GeminiQAService(embedding_service=self.emb)

    def test_multi_citation_formatting(self):
        """Failure D test: Multi-citations like [S1, S2] become structured [Source: ...]."""
        self.mock_client.models.generate_content.return_value = mock_genai_reply({
            "answerable": True,
            "answer": "Microcontrollers and microcomputers are distinct [S1, S2].",
            "used_snippets": [1, 2],
        })
        res = self.svc().answer_question("Distinguish components")
        self.assertTrue(res.grounded)
        self.assertNotIn("[S1, S2]", res.answer)
        self.assertIn("[Source: GENERIC _UNIT 1.pdf — Page 1, Page 2]", res.answer)

    def test_intel_8051_bit_width_correction(self):
        """Failure C test: OCR typo 'l8-bit' must not result in '18-bit microcontroller'."""
        self.mock_client.models.generate_content.return_value = mock_genai_reply({
            "answerable": True,
            "answer": "The Intel 8051 was introduced in 1980 as an 18-bit microcontroller [S1].",
            "used_snippets": [1],
        })
        res = self.svc().answer_question("What is the Intel 8051?")
        self.assertTrue(res.grounded)
        self.assertNotIn("18-bit microcontroller", res.answer)
        self.assertIn("8-bit microcontroller", res.answer)

    def test_definition_mode_with_simple_words(self):
        """Failure B test: Faithful technical definition + beginner explanation."""
        definition_answer = (
            "A microcomputer is a small, relatively inexpensive computer with a microprocessor "
            "as its CPU, incorporating memory and I/O facilities on a printed circuit board [S2].\n\n"
            "**In simple words:** A complete computer built on a single board where a microprocessor "
            "acts as the brain."
        )
        self.mock_client.models.generate_content.return_value = mock_genai_reply({
            "answerable": True,
            "answer": definition_answer,
            "used_snippets": [2],
        })
        res = self.svc().answer_question(
            "What is the definition of a microcomputer in this document? Also explain in simple words and provide the source."
        )
        self.assertTrue(res.grounded)
        self.assertIn("printed circuit board", res.answer)
        self.assertIn("**In simple words:**", res.answer)
        self.assertIn("[Source: GENERIC _UNIT 1.pdf — Page 2]", res.answer)

    def test_document_wide_summary_routing(self):
        """Failure A test: Whole-document summary routes to DocumentSummarizer and covers all pages."""
        with patch.object(DocumentSummarizer, "summarize") as mock_summarize:
            mock_summarize.return_value = {
                "answer": "### Document Overview\nThis unit covers microcomputers and microcontrollers across all 9 pages [Source: GENERIC _UNIT 1.pdf — Page 1, Page 2, Page 6].",
                "grounded": True,
                "sources": [
                    {"document_id": self.doc.id, "document_title": self.doc.title, "page_number": 1},
                    {"document_id": self.doc.id, "document_title": self.doc.title, "page_number": 2},
                    {"document_id": self.doc.id, "document_title": self.doc.title, "page_number": 6},
                ],
            }
            res = self.svc().answer_question(
                "Analyze all 9 pages of this document and identify all topics",
                document_id=self.doc.id,
            )
            self.assertTrue(res.grounded)
            self.assertIn("Document Overview", res.answer)
            self.assertEqual(len(res.sources), 3)
            self.assertIsNotNone(res.diagnostics)
            self.assertEqual(res.diagnostics["intent"], "document_summary")
            self.assertEqual(res.diagnostics["retrieval_strategy"], "document_wide_synthesis")
