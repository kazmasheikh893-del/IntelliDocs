import hashlib
import os

from django import forms
from django.conf import settings

from .models import Document


class DocumentUploadForm(forms.ModelForm):
    """Validates extension, size, and basic content signature of uploads."""

    class Meta:
        model = Document
        fields = ["file"]
        labels = {"file": "Document (PDF or TXT)"}
        widgets = {
            "file": forms.ClearableFileInput(
                attrs={"class": "file-input", "accept": ".pdf,.txt", "id": "file-upload"}
            )
        }

    def clean_file(self):
        f = self.cleaned_data.get("file")
        if not f:
            raise forms.ValidationError("Please select a file to upload.")

        ext = os.path.splitext(os.path.basename(f.name))[1].lower()
        allowed = getattr(settings, "ALLOWED_FILE_EXTENSIONS", [".pdf", ".txt"])
        if ext not in allowed:
            raise forms.ValidationError(
                f"Unsupported file format '{ext or 'unknown'}'. Allowed formats: {', '.join(allowed)}."
            )
        if f.size == 0:
            raise forms.ValidationError("The uploaded file is empty (0 bytes).")
        max_size = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
        if f.size > max_size:
            raise forms.ValidationError(
                f"File size exceeds the maximum allowed size of {max_size // (1024 * 1024)} MB."
            )

        head = f.read(1024)
        f.seek(0)
        if ext == ".pdf" and not head.lstrip().startswith(b"%PDF-"):
            raise forms.ValidationError("This file is not a valid PDF (missing PDF header).")
        if ext == ".txt" and b"\x00" in head:
            raise forms.ValidationError("This file does not look like plain text (binary content detected).")

        digest = hashlib.sha256()
        for block in f.chunks():
            digest.update(block)
        f.seek(0)
        self.content_hash = digest.hexdigest()
        if Document.objects.filter(content_hash=self.content_hash,
                                   status__in=[Document.STATUS_READY, Document.STATUS_PROCESSING]).exists():
            raise forms.ValidationError("This exact file has already been uploaded.")
        return f
