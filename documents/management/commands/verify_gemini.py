"""
Live connectivity check against the real Gemini API.

    python manage.py verify_gemini

Never prints the API key. Lists models available to the key, confirms the configured models
support the required operations, then makes one real embedding and one real generation call.
"""

import time

from django.conf import settings
from django.core.management.base import BaseCommand
from google import genai
from google.genai import types

from documents.services.embeddings import PLACEHOLDER_KEYS
from documents.services.errors import translate_exception


def _short(name: str) -> str:
    return name.split("/", 1)[-1]


class Command(BaseCommand):
    help = "Verify Gemini credentials, configured models, and real embedding/generation calls."

    def handle(self, *args, **options):
        key = (settings.GEMINI_API_KEY or "").strip()
        if key in PLACEHOLDER_KEYS:
            self.stdout.write(self.style.ERROR(
                "[BLOCKED] GEMINI_API_KEY is not configured. Copy .env.example to .env and set "
                "your key (https://aistudio.google.com/). Live verification NOT performed."))
            return

        client = genai.Client(api_key=key)
        emb_model = settings.GEMINI_EMBEDDING_MODEL
        gen_model = settings.GEMINI_GENERATION_MODEL
        dim = settings.GEMINI_EMBEDDING_DIM
        self.stdout.write("API key: configured (value hidden)")

        try:
            available = {_short(m.name): (m.supported_actions or []) for m in client.models.list()}
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"[FAIL] Could not list models: {translate_exception(exc).user_message}"))
            return
        self.stdout.write(f"Models visible to this key: {len(available)}")

        for label, model, action in (("embedding", emb_model, "embedContent"),
                                     ("generation", gen_model, "generateContent")):
            actions = available.get(model)
            if actions is None:
                self.stdout.write(self.style.ERROR(f"[FAIL] {label} model '{model}' is not available to this key."))
                suggestions = sorted(n for n, a in available.items() if action in a)[:15]
                self.stdout.write(f"       Models supporting {action}: {', '.join(suggestions)}")
                return
            if action not in actions:
                self.stdout.write(self.style.ERROR(f"[FAIL] '{model}' does not support {action}."))
                return
            self.stdout.write(self.style.SUCCESS(f"[OK] {label} model '{model}' supports {action}"))

        try:
            t = time.time()
            r = client.models.embed_content(
                model=emb_model, contents="IntelliDocs connectivity test.",
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dim))
            got = len(r.embeddings[0].values)
            ok = got == dim
            style = self.style.SUCCESS if ok else self.style.ERROR
            self.stdout.write(style(f"[{'OK' if ok else 'FAIL'}] live embedding: {got} dims (expected {dim}), "
                                    f"{(time.time() - t) * 1000:.0f} ms"))
            if not ok:
                return
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"[FAIL] live embedding: {translate_exception(exc).user_message}"))
            return

        try:
            t = time.time()
            r = client.models.generate_content(model=gen_model, contents="Reply with the single word: READY")
            self.stdout.write(self.style.SUCCESS(
                f"[OK] live generation returned text ({len(r.text or '')} chars), {(time.time() - t) * 1000:.0f} ms"))
        except Exception as exc:
            code = getattr(exc, "code", None)
            detail = f" (HTTP {code})" if code else ""
            self.stdout.write(self.style.ERROR(f"[FAIL] live generation{detail}: {translate_exception(exc).user_message}"))
            return

        self.stdout.write(self.style.SUCCESS("All live Gemini checks passed."))
