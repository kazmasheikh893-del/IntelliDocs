import io
import os
import tempfile
import shutil

from django.test import SimpleTestCase

from documents.services.extractor import DocumentExtractor, ExtractionError
from .helpers import make_pdf, make_blank_pdf


class DocumentExtractorTests(SimpleTestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_txt_utf8(self):
        path = self._write("a.txt", "Héllo wörld\nsecond line".encode("utf-8"))
        pages = DocumentExtractor.extract(path)
        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0]["page_number"])
        self.assertIn("Héllo wörld", pages[0]["text"])

    def test_txt_utf8_bom_is_stripped(self):
        path = self._write("a.txt", b"\xef\xbb\xbfBOM text")
        self.assertEqual(DocumentExtractor.extract(path)[0]["text"], "BOM text")

    def test_txt_latin1_fallback(self):
        path = self._write("a.txt", "café".encode("latin-1"))
        self.assertIn("caf", DocumentExtractor.extract(path)[0]["text"])

    def test_empty_txt_rejected(self):
        path = self._write("e.txt", b"   \n\t ")
        with self.assertRaisesMessage(ExtractionError, "empty"):
            DocumentExtractor.extract(path)

    def test_pdf_page_mapping_is_one_based(self):
        path = self._write("two.pdf", make_pdf(["Alpha fact on first page", "Bravo fact on second page"]))
        pages = DocumentExtractor.extract(path)
        self.assertEqual([p["page_number"] for p in pages], [1, 2])
        self.assertIn("Alpha", pages[0]["text"])
        self.assertIn("Bravo", pages[1]["text"])
        self.assertNotIn("Bravo", pages[0]["text"])

    def test_scanned_or_blank_pdf_rejected_with_ocr_note(self):
        path = self._write("blank.pdf", make_blank_pdf(2))
        with self.assertRaises(ExtractionError) as ctx:
            DocumentExtractor.extract(path)
        self.assertIn("readable text", str(ctx.exception))

    def test_corrupt_pdf_rejected(self):
        path = self._write("bad.pdf", b"%PDF-1.4 this is not really a pdf")
        with self.assertRaises(ExtractionError):
            DocumentExtractor.extract(path)

    def test_unsupported_extension(self):
        path = self._write("r.docx", b"x")
        with self.assertRaisesMessage(ExtractionError, "Unsupported"):
            DocumentExtractor.extract(path)

    def test_missing_file(self):
        with self.assertRaises(ExtractionError):
            DocumentExtractor.extract(os.path.join(self.dir, "nope.txt"))
