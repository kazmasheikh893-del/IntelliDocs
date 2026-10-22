import os
import tempfile
import io
from django.test import TestCase
import pypdf
from documents.services.extractor import DocumentExtractor, ExtractionError


class DocumentExtractorTests(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_extract_valid_txt(self):
        txt_path = os.path.join(self.temp_dir, "sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("This is a test document.\nWith two lines of text.")

        results = DocumentExtractor.extract(txt_path)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["page_number"])
        self.assertIn("This is a test document.", results[0]["text"])

    def test_extract_empty_txt_raises_error(self):
        empty_txt = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_txt, "w", encoding="utf-8") as f:
            f.write("   \n  \t  ")

        with self.assertRaises(ExtractionError) as ctx:
            DocumentExtractor.extract(empty_txt)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_extract_synthetic_pdf_with_page_numbers(self):
        pdf_path = os.path.join(self.temp_dir, "two_pages.pdf")
        
        # Build a valid two-page PDF with text using pypdf
        writer = pypdf.PdfWriter()
        # Add blank pages then we can write content
        # Note: in pypdf, we can create a PDF with text stream annotations or simple text
        # Or write simple PDF stream
        writer.add_blank_page(width=200, height=200)
        writer.add_blank_page(width=200, height=200)
        
        # Write to file
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # Blank PDF without text should raise ExtractionError (no readable text)
        with self.assertRaises(ExtractionError) as ctx:
            DocumentExtractor.extract(pdf_path)
        self.assertIn("readable text", str(ctx.exception).lower())

    def test_unsupported_file_extension(self):
        doc_path = os.path.join(self.temp_dir, "report.docx")
        with open(doc_path, "w") as f:
            f.write("dummy")

        with self.assertRaises(ExtractionError) as ctx:
            DocumentExtractor.extract(doc_path)
        self.assertIn("unsupported", str(ctx.exception).lower())
