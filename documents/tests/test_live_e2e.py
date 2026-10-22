"""
LIVE end-to-end acceptance test against the real Gemini API.

Skipped unless BOTH are true:
  * GEMINI_API_KEY is configured (locally, via .env)
  * RUN_LIVE_TESTS=1 is set in the environment
It uses a synthetic PDF and an isolated test database, and consumes real API quota.

    $env:RUN_LIVE_TESTS = "1"
    .\\.venv\\Scripts\\python.exe manage.py test documents.tests.test_live_e2e -v 2
"""

import json
import os
import shutil
import tempfile
import unittest

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import Document
from documents.services.generator import INSUFFICIENT_EVIDENCE_ANSWER
from .helpers import make_pdf

LIVE = os.environ.get("RUN_LIVE_TESTS") == "1" and bool(settings.GEMINI_API_KEY.strip())
TMP_MEDIA = tempfile.mkdtemp(prefix="intellidocs-live-")


def tearDownModule():
    shutil.rmtree(TMP_MEDIA, ignore_errors=True)


@unittest.skipUnless(LIVE, "Live test: set RUN_LIVE_TESTS=1 and configure GEMINI_API_KEY to run.")
@override_settings(MEDIA_ROOT=TMP_MEDIA, RAG_MIN_SIMILARITY=0.3)
class LiveEndToEndTests(TestCase):
    def test_upload_ask_cite_and_fallback(self):
        pdf = make_pdf([
            "The Zorblax-9 reactor was commissioned in the year 1987 in the city of Quillham.",
            "The Zorblax-9 reactor has a maximum output of 450 megawatts and uses graphite moderation.",
        ])
        r = self.client.post(reverse("documents:upload"),
                             {"file": SimpleUploadedFile("zorblax.pdf", pdf)}, follow=True)
        doc = Document.objects.get(title="zorblax.pdf")
        self.assertEqual(doc.status, Document.STATUS_READY, doc.error_message)
        self.assertEqual(doc.page_count, 2)
        self.assertGreaterEqual(doc.chunks.count(), 2)

        url = reverse("documents:api_query")

        def ask(q, **extra):
            resp = self.client.post(url, data=json.dumps({"question": q, **extra}),
                                    content_type="application/json")
            self.assertEqual(resp.status_code, 200, resp.content)
            return resp.json()

        out = ask("What is the maximum output of the Zorblax-9 reactor?")
        self.assertTrue(out["grounded"], out)
        self.assertIn("450", out["answer"])
        self.assertTrue(any(s["page_number"] == 2 and s["document_title"] == "zorblax.pdf"
                            for s in out["sources"]), out["sources"])

        out = ask("In which year was the Zorblax-9 reactor commissioned?", document_id=doc.id)
        self.assertIn("1987", out["answer"])
        self.assertTrue(any(s["page_number"] == 1 for s in out["sources"]))

        out = ask("Who won the 1998 football World Cup final?")
        self.assertEqual(out["answer"], INSUFFICIENT_EVIDENCE_ANSWER)
        self.assertEqual(out["sources"], [])
