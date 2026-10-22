from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from documents.forms import DocumentUploadForm


class DocumentUploadFormTests(TestCase):
    def test_valid_txt_upload(self):
        txt_file = SimpleUploadedFile("sample.txt", b"Valid text file content.", content_type="text/plain")
        form = DocumentUploadForm(data={}, files={"file": txt_file})
        self.assertTrue(form.is_valid())

    def test_valid_pdf_upload(self):
        pdf_file = SimpleUploadedFile("sample.pdf", b"%PDF-1.4 sample content", content_type="application/pdf")
        form = DocumentUploadForm(data={}, files={"file": pdf_file})
        self.assertTrue(form.is_valid())

    def test_disallowed_extension(self):
        exe_file = SimpleUploadedFile("sample.exe", b"binary content", content_type="application/x-msdownload")
        form = DocumentUploadForm(data={}, files={"file": exe_file})
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)
        self.assertIn("Unsupported file format", form.errors["file"][0])

    def test_empty_file_rejected(self):
        empty_file = SimpleUploadedFile("empty.txt", b"", content_type="text/plain")
        form = DocumentUploadForm(data={}, files={"file": empty_file})
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)
