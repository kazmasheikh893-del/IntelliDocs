"""
Document text extraction service for PDF and TXT files.
Preserves page-level metadata for PDFs.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pypdf


class ExtractionError(Exception):
    """Custom exception raised when document extraction fails or produces invalid content."""
    pass


class DocumentExtractor:
    """Extracts text content and page-level metadata from supported document files."""

    @classmethod
    def extract(cls, file_path: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extracts text from a given file path.
        
        Returns:
            List of dicts: [
                {"page_number": int | None, "text": str},
                ...
            ]
        """
        if not os.path.exists(file_path):
            raise ExtractionError(f"File not found: {file_path}")

        name = filename or file_path
        ext = os.path.splitext(name)[1].lower()

        if ext == ".pdf":
            return cls._extract_pdf(file_path)
        elif ext == ".txt":
            return cls._extract_txt(file_path)
        else:
            raise ExtractionError(f"Unsupported file format: {ext}. Only .pdf and .txt are supported.")

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Cleans private-use characters, OCR font glitches, and normalizes spacing."""
        if not text:
            return ""
        # 1. Replace private use Unicode characters (often bullet points in fonts) with standard bullet
        t = re.sub(r"[\ue000-\uf8ff]", "• ", text)
        # 2. Fix OCR glitch where letter 'l' or 'I' is glued immediately before a bit-width number (e.g. 'l8-bit' -> '8-bit')
        t = re.sub(r"\b[lI](\d+)\s*[-–]?\s*bit\b", r"\1-bit", t, flags=re.IGNORECASE)
        # 3. Clean null bytes or rogue non-printable control chars (preserve \n, \t, \r)
        t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
        # 4. Standardize CRLF to LF
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        # 5. Clean trailing whitespace on lines
        lines = [line.strip() for line in t.split("\n")]
        return "\n".join(lines).strip()

    @classmethod
    def _extract_pdf(cls, file_path: str) -> List[Dict[str, Any]]:
        pages_data: List[Dict[str, Any]] = []

        try:
            reader = pypdf.PdfReader(file_path)
        except Exception as e:
            raise ExtractionError(f"Could not open or parse PDF file: {str(e)}")

        if reader.is_encrypted:
            try:
                # Try decrypting with empty password
                reader.decrypt("")
            except Exception:
                raise ExtractionError("PDF is password-protected or encrypted. Please upload an unencrypted document.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ExtractionError("The PDF document contains 0 pages.")

        has_any_text = False
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""
            
            cleaned_text = cls._clean_text(raw_text)
            if cleaned_text:
                has_any_text = True
            
            pages_data.append({
                "page_number": page_num,
                "text": cleaned_text
            })

        if not has_any_text:
            raise ExtractionError(
                "No readable text could be extracted from this PDF. "
                "The file may be a scanned image or contain non-extractable elements."
            )

        return pages_data

    @classmethod
    def _extract_txt(cls, file_path: str) -> List[Dict[str, Any]]:
        encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
        content = None

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        for enc in encodings:
            try:
                content = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ExtractionError("Unable to decode text file with standard encodings (UTF-8, Latin-1).")

        cleaned_text = cls._clean_text(content)
        if not cleaned_text:
            raise ExtractionError("The uploaded text file is empty.")

        return [{
            "page_number": None,
            "text": cleaned_text
        }]
