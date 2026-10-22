from django.test import SimpleTestCase

from documents.services.chunker import TextChunker


class TextChunkerTests(SimpleTestCase):
    def test_preserves_page_numbers_and_sequential_index(self):
        pages = [
            {"page_number": 1, "text": "Page one text."},
            {"page_number": 2, "text": "Page two text."},
            {"page_number": 3, "text": "Page three text."},
        ]
        chunks = TextChunker(500, 50).chunk_pages(pages)
        self.assertEqual([c.page_number for c in chunks], [1, 2, 3])
        self.assertEqual([c.chunk_index for c in chunks], [0, 1, 2])

    def test_txt_keeps_null_page(self):
        chunks = TextChunker(100, 20).chunk_pages([{"page_number": None, "text": "Plain text."}])
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].page_number)

    def test_chunks_never_exceed_size_and_overlap(self):
        text = " ".join(f"Sentence number {i} says something useful." for i in range(60))
        size, overlap = 200, 40
        chunks = TextChunker(size, overlap).chunk_pages([{"page_number": 4, "text": text}])
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(len(c.content), size)
            self.assertEqual(c.page_number, 4)
        # consecutive chunks share some text (overlap)
        shared = [set(a.content.split()) & set(b.content.split())
                  for a, b in zip(chunks, chunks[1:])]
        self.assertTrue(all(shared))

    def test_no_text_lost(self):
        words = [f"w{i}" for i in range(400)]
        chunks = TextChunker(120, 20).chunk_pages([{"page_number": 1, "text": " ".join(words)}])
        joined = " ".join(c.content for c in chunks)
        for w in words:
            self.assertIn(w, joined)

    def test_text_without_separators_is_still_split(self):
        chunks = TextChunker(50, 10).chunk_pages([{"page_number": 1, "text": "x" * 500}])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.content) <= 50 for c in chunks))

    def test_empty_and_whitespace_pages_skipped(self):
        chunks = TextChunker(100, 10).chunk_pages(
            [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "  \n "},
             {"page_number": 3, "text": "Real content"}])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 3)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(TextChunker(100, 10).chunk_pages([]), [])

    def test_invalid_parameters(self):
        for size, overlap in ((50, 50), (40, 50)):
            with self.assertRaises(ValueError):
                TextChunker(size, overlap)
