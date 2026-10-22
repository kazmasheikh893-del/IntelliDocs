from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings
from google.genai import errors as genai_errors

from documents.services.embeddings import GeminiEmbeddingService, normalize
from documents.services.errors import (
    EmbeddingError, InvalidCredentialsError, MissingApiKeyError, ModelUnavailableError,
    QuotaExceededError, GeminiNetworkError, translate_exception,
)


def emb(values):
    return SimpleNamespace(values=values)


class EmbeddingServiceTests(SimpleTestCase):
    @override_settings(GEMINI_API_KEY="")
    def test_missing_key(self):
        with self.assertRaises(MissingApiKeyError):
            GeminiEmbeddingService()

    @override_settings(GEMINI_API_KEY="your_gemini_api_key_here")
    def test_placeholder_key(self):
        with self.assertRaises(MissingApiKeyError):
            GeminiEmbeddingService()

    @override_settings(GEMINI_API_KEY="k", GEMINI_EMBEDDING_MODEL="m", GEMINI_EMBEDDING_DIM=3)
    @patch("documents.services.embeddings.genai.Client")
    def test_vectors_are_normalised_and_request_is_configured(self, client_cls):
        client = client_cls.return_value
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[emb([3.0, 0.0, 4.0])])
        vec = GeminiEmbeddingService().embed_text("hello")
        self.assertAlmostEqual(sum(x * x for x in vec) ** 0.5, 1.0, places=9)
        kwargs = client.models.embed_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "m")
        self.assertEqual(kwargs["config"].output_dimensionality, 3)
        self.assertEqual(kwargs["config"].task_type, "RETRIEVAL_QUERY")

    @override_settings(GEMINI_API_KEY="k", GEMINI_EMBEDDING_DIM=3)
    @patch("documents.services.embeddings.genai.Client")
    def test_batching_and_document_task_type(self, client_cls):
        client = client_cls.return_value
        client.models.embed_content.side_effect = lambda **kw: SimpleNamespace(
            embeddings=[emb([1.0, 1.0, 1.0]) for _ in kw["contents"]])
        out = GeminiEmbeddingService().embed_batch([f"t{i}" for i in range(120)])
        self.assertEqual(len(out), 120)
        self.assertEqual(client.models.embed_content.call_count, 3)  # 50 + 50 + 20
        self.assertEqual(client.models.embed_content.call_args.kwargs["config"].task_type, "RETRIEVAL_DOCUMENT")

    @override_settings(GEMINI_API_KEY="k", GEMINI_EMBEDDING_DIM=3)
    @patch("documents.services.embeddings.genai.Client")
    def test_dimension_mismatch_detected(self, client_cls):
        client_cls.return_value.models.embed_content.return_value = SimpleNamespace(embeddings=[emb([1.0, 2.0])])
        with self.assertRaises(EmbeddingError):
            GeminiEmbeddingService().embed_text("x")

    @override_settings(GEMINI_API_KEY="k", GEMINI_EMBEDDING_DIM=3)
    @patch("documents.services.embeddings.genai.Client")
    def test_count_mismatch_detected(self, client_cls):
        client_cls.return_value.models.embed_content.return_value = SimpleNamespace(embeddings=[])
        with self.assertRaises(EmbeddingError):
            GeminiEmbeddingService().embed_batch(["a"])

    @override_settings(GEMINI_API_KEY="k")
    @patch("documents.services.embeddings.genai.Client")
    def test_empty_text_rejected(self, _):
        with self.assertRaises(ValueError):
            GeminiEmbeddingService().embed_text("   ")

    def test_zero_vector_rejected(self):
        with self.assertRaises(EmbeddingError):
            normalize([0.0, 0.0])

    @override_settings(GEMINI_API_KEY="SECRET-KEY-VALUE-123", GEMINI_EMBEDDING_DIM=3)
    @patch("documents.services.embeddings.genai.Client")
    def test_api_errors_are_translated_without_leaking_secrets(self, client_cls):
        err = genai_errors.APIError(429, {"error": {"message": "quota SECRET-KEY-VALUE-123"}})
        client_cls.return_value.models.embed_content.side_effect = err
        with self.assertRaises(QuotaExceededError) as ctx:
            GeminiEmbeddingService().embed_text("x")
        self.assertNotIn("SECRET-KEY-VALUE-123", ctx.exception.user_message)


class ErrorTranslationTests(SimpleTestCase):
    def _api(self, code, msg="boom"):
        return genai_errors.APIError(code, {"error": {"message": msg}})

    def test_mapping(self):
        self.assertIsInstance(translate_exception(self._api(429)), QuotaExceededError)
        self.assertIsInstance(translate_exception(self._api(401)), InvalidCredentialsError)
        self.assertIsInstance(translate_exception(self._api(403)), InvalidCredentialsError)
        self.assertIsInstance(translate_exception(self._api(400, "API key not valid")), InvalidCredentialsError)
        self.assertIsInstance(translate_exception(self._api(404)), ModelUnavailableError)
        self.assertIsInstance(translate_exception(ConnectionError("x")), GeminiNetworkError)
        self.assertIn("500", translate_exception(self._api(500)).user_message)
