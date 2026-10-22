from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from documents.forms import DocumentUploadForm
from documents.models import Document
from .helpers import make_pdf


def form_for(name, data, content_type="application/octet-stream"):
    return DocumentUploadForm(data={}, files={"file": SimpleUploadedFile(name, data, content_type=content_type)})


class DocumentUploadFormTests(TestCase):
    def test_valid_txt(self):
        self.assertTrue(form_for("sample.txt", b"Valid text file content.").is_valid())

    def test_valid_real_pdf(self):
        self.assertTrue(form_for("sample.pdf", make_pdf(["hello"])).is_valid())

    def test_extension_check_is_case_insensitive(self):
        self.assertTrue(form_for("UPPER.TXT", b"text").is_valid())

    def test_disallowed_extension(self):
        f = form_for("sample.exe", b"binary")
        self.assertFalse(f.is_valid())
        self.assertIn("Unsupported file format", f.errors["file"][0])

    def test_double_extension_trick_rejected(self):
        self.assertFalse(form_for("report.pdf.exe", b"MZ").is_valid())

    def test_empty_file_rejected(self):
        self.assertFalse(form_for("empty.txt", b"").is_valid())

    def test_pdf_extension_with_non_pdf_content_rejected(self):
        f = form_for("fake.pdf", b"<html>not a pdf</html>")
        self.assertFalse(f.is_valid())
        self.assertIn("not a valid PDF", f.errors["file"][0])

    def test_binary_txt_rejected(self):
        self.assertFalse(form_for("bin.txt", b"ab\x00cd").is_valid())

    @override_settings(MAX_UPLOAD_SIZE_BYTES=10)
    def test_oversize_rejected(self):
        f = form_for("big.txt", b"x" * 11)
        self.assertFalse(f.is_valid())
        self.assertIn("exceeds", f.errors["file"][0])

    def test_duplicate_of_ready_document_rejected_but_failed_allowed(self):
        f = form_for("a.txt", b"same bytes")
        self.assertTrue(f.is_valid())
        digest = f.content_hash
        failed = Document.objects.create(title="a.txt", file_type="txt", content_hash=digest,
                                         status=Document.STATUS_FAILED)
        self.assertTrue(form_for("b.txt", b"same bytes").is_valid())
        failed.status = Document.STATUS_READY
        failed.save()
        self.assertFalse(form_for("c.txt", b"same bytes").is_valid())
