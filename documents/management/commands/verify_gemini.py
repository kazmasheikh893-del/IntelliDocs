"""
Django management command to test and verify official Google Gemini API connectivity.
Usage: python manage.py verify_gemini
"""

import time
from django.core.management.base import BaseCommand
from django.conf import settings
from google import genai
from google.genai.errors import APIError


class Command(BaseCommand):
    help = "Verifies real connectivity, credentials, and model availability for the Google Gemini API."

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("==> Testing Google Gemini API Configuration..."))

        api_key = getattr(settings, "GEMINI_API_KEY", "")
        if not api_key:
            self.stdout.write(
                self.style.ERROR(
                    "[FAIL] GEMINI_API_KEY is not set in your environment or .env file.\n"
                    "-> Action required: Open .env and set GEMINI_API_KEY=your_key_here\n"
                    "   You can obtain a free key from Google AI Studio: https://aistudio.google.com/"
                )
            )
            return

        if api_key == "your_gemini_api_key_here":
            self.stdout.write(
                self.style.ERROR(
                    "[FAIL] GEMINI_API_KEY contains the example placeholder value.\n"
                    "-> Action required: Replace the placeholder in .env with your real Google Gemini API key."
                )
            )
            return

        # Obfuscate key in output
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        self.stdout.write(self.style.SUCCESS(f"[OK] Found GEMINI_API_KEY ({masked_key})"))

        client = genai.Client(api_key=api_key)
        embedding_model = getattr(settings, "GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        generation_model = getattr(settings, "GEMINI_GENERATION_MODEL", "gemini-2.0-flash")

        # 1. Test Embedding Model
        self.stdout.write(f"--> Testing Embedding Model: {embedding_model} ...")
        start_time = time.time()
        try:
            emb_resp = client.models.embed_content(
                model=embedding_model,
                contents="IntelliDocs RAG embedding connectivity test.",
            )
            elapsed = (time.time() - start_time) * 1000
            if emb_resp.embeddings and len(emb_resp.embeddings[0].values) > 0:
                dims = len(emb_resp.embeddings[0].values)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"[OK] Embedding SUCCESS ({dims} dimensions, latency: {elapsed:.1f}ms)"
                    )
                )
            else:
                self.stdout.write(self.style.WARNING("[WARN] Embedding returned empty values."))
        except APIError as e:
            self.stdout.write(
                self.style.ERROR(f"[FAIL] Embedding API Error: {e.message} (HTTP code: {e.code})")
            )
            return
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[FAIL] Embedding Unexpected Error: {str(e)}"))
            return

        # 2. Test Generation Model
        self.stdout.write(f"--> Testing Generation Model: {generation_model} ...")
        start_time = time.time()
        try:
            gen_resp = client.models.generate_content(
                model=generation_model,
                contents="Reply with the single word: READY",
            )
            elapsed = (time.time() - start_time) * 1000
            resp_text = (gen_resp.text or "").strip()
            self.stdout.write(
                self.style.SUCCESS(
                    f"[OK] Generation SUCCESS (Response: '{resp_text}', latency: {elapsed:.1f}ms)"
                )
            )
        except APIError as e:
            self.stdout.write(
                self.style.ERROR(f"[FAIL] Generation API Error: {e.message} (HTTP code: {e.code})")
            )
            return
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[FAIL] Generation Unexpected Error: {str(e)}"))
            return

        self.stdout.write(
            self.style.SUCCESS(
                "\n[SUCCESS] All Gemini API checks passed successfully! Your RAG assistant is ready."
            )
        )
