import os
from django import forms
from django.conf import settings
from .models import Document


class DocumentUploadForm(forms.ModelForm):
    """Form for validating and uploading PDF and TXT documents."""

    class Meta:
        model = Document
        fields = ["file"]
        widgets = {
            "file": forms.ClearableFileInput(
                attrs={
                    "class": "file-input",
                    "accept": ".pdf,.txt,application/pdf,text/plain",
                    "id": "file-upload",
                }
            )
        }

    def clean_file(self):
        uploaded_file = self.cleaned_data.get("file")
        if not uploaded_file:
            raise forms.ValidationError("Please select a file to upload.")

        # Check file extension
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        allowed_extensions = getattr(settings, "ALLOWED_FILE_EXTENSIONS", [".pdf", ".txt"])
        if ext not in allowed_extensions:
            raise forms.ValidationError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_extensions)}"
            )

        # Check file size limit
        max_size = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
        if uploaded_file.size > max_size:
            max_mb = max_size / (1024 * 1024)
            raise forms.ValidationError(f"File size exceeds maximum allowed limit of {max_mb:.0f} MB.")

        # Check non-empty upload
        if uploaded_file.size == 0:
            raise forms.ValidationError("Uploaded file is empty (0 bytes).")

        return uploaded_file
