from django.test import TestCase
from documents.services.chunker import TextChunker, TextChunk


class TextChunkerTests(TestCase):
    def test_chunk_preserves_page_numbers(self):
        pages_data = [
            {"page_number": 1, "text": "Page one text content about artificial intelligence."},
            {"page_number": 2, "text": "Page two text content about retrieval augmented generation."},
            {"page_number": 3, "text": "Page three text content about vector databases and citations."},
        ]

        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_pages(pages_data)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertIn("Page one", chunks[0].content)

        self.assertEqual(chunks[1].page_number, 2)
        self.assertEqual(chunks[1].chunk_index, 1)
        self.assertIn("Page two", chunks[1].content)

        self.assertEqual(chunks[2].page_number, 3)
        self.assertEqual(chunks[2].chunk_index, 2)
        self.assertIn("Page three", chunks[2].content)

    def test_txt_preserves_null_page_number(self):
        pages_data = [
            {"page_number": None, "text": "A standard text document without page numbers."}
        ]

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_pages(pages_data)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].page_number)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_chunk_splitting_respects_max_size_and_overlap(self):
        # Paragraph with 300 characters
        long_paragraph = "Python is an interpreted, high-level, general-purpose programming language. " * 5
        pages_data = [{"page_number": 1, "text": long_paragraph}]

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_pages(pages_data)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.page_number, 1)
            self.assertLessEqual(len(chunk.content), 120)  # within reasonable boundary

    def test_invalid_parameters_raise_error(self):
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=50, chunk_overlap=50)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=40, chunk_overlap=50)
