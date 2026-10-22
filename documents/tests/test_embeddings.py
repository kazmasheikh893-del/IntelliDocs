from unittest.mock import patch, MagicMock
from django.test import TestCase, override_settings
from documents.services.embeddings import (
    GeminiEmbeddingService,
    MissingApiKeyError,
    EmbeddingError,
)


class GeminiEmbeddingServiceTests(TestCase):
    @override_settings(GEMINI_API_KEY="")
    def test_missing_api_key_raises_exception(self):
        with self.assertRaises(MissingApiKeyError):
            GeminiEmbeddingService()

    @override_settings(GEMINI_API_KEY="your_gemini_api_key_here")
    def test_placeholder_api_key_raises_exception(self):
        with self.assertRaises(MissingApiKeyError):
            GeminiEmbeddingService()

    @override_settings(GEMINI_API_KEY="test_api_key_12345")
    @patch("documents.services.embeddings.genai.Client")
    def test_embed_text_calls_client_properly(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client.models.embed_content.return_value = mock_response

        service = GeminiEmbeddingService()
        vector = service.embed_text("Sample text query")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        mock_client.models.embed_content.assert_called_once()
