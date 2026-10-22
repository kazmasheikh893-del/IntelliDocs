"""Test helpers: build small real PDFs (with extractable text) without extra dependencies."""


def make_pdf(pages):
    """Return bytes of a valid PDF with one text line per page (Helvetica)."""
    objs = []  # index i holds object number i+1
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET".encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {page_ids[i] + 1} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_blank_pdf(num_pages=1):
    """A valid PDF with no text (stands in for a scanned / image-only PDF)."""
    import io
    import pypdf
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeEmbeddingService:
    """Deterministic offline embedder for UNIT TESTS ONLY (never used by the application)."""

    def __init__(self, model="test-embed", dimensions=8, vectors=None):
        self.model = model
        self.dimensions = dimensions
        self._vectors = vectors or {}
        self.fail_with = None

    def _vec(self, text):
        for key, vec in self._vectors.items():
            if key.lower() in text.lower():
                return vec
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        v = [b + 1 for b in h[: self.dimensions]]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    def embed_text(self, text):
        if self.fail_with:
            raise self.fail_with
        return self._vec(text)

    def embed_batch(self, texts):
        if self.fail_with:
            raise self.fail_with
        return [self._vec(t) for t in texts]

