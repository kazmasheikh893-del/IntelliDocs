"""Safe, typed errors for the Gemini integration. Messages never contain credentials."""

from typing import Optional


class GeminiError(Exception):
    """Base class. `user_message` is always safe to display to end users."""

    http_status = 502
    user_message = "The AI service returned an error. Please try again."

    def __init__(self, user_message: Optional[str] = None):
        if user_message:
            self.user_message = user_message
        super().__init__(self.user_message)


class MissingApiKeyError(GeminiError):
    http_status = 503
    user_message = (
        "GEMINI_API_KEY is not configured. Add your key to the local .env file "
        "(see .env.example) and restart the server."
    )


class InvalidCredentialsError(GeminiError):
    http_status = 503
    user_message = "The Gemini API rejected the configured API key. Check GEMINI_API_KEY."


class ModelUnavailableError(GeminiError):
    http_status = 503
    user_message = (
        "The configured Gemini model is unavailable. Run 'python manage.py verify_gemini' "
        "and update GEMINI_EMBEDDING_MODEL / GEMINI_GENERATION_MODEL."
    )


class QuotaExceededError(GeminiError):
    http_status = 429
    user_message = "The Gemini API quota or rate limit was reached. Please wait and try again."


class GeminiNetworkError(GeminiError):
    http_status = 502
    user_message = "Could not reach the Gemini API. Check your network connection and try again."


class EmbeddingError(GeminiError):
    pass


class IncompatibleEmbeddingsError(GeminiError):
    http_status = 409
    user_message = (
        "Stored document vectors were created with a different embedding model or dimension. "
        "Run 'python manage.py reembed' to rebuild them."
    )


def translate_exception(exc: Exception) -> GeminiError:
    """Convert any SDK / network exception into a safe GeminiError."""
    if isinstance(exc, GeminiError):
        return exc
    try:
        from google.genai import errors as genai_errors
    except Exception:  # pragma: no cover
        genai_errors = None

    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code == 429:
            return QuotaExceededError()
        if code in (401, 403):
            return InvalidCredentialsError()
        if code == 400 and "api key" in str(exc).lower():
            return InvalidCredentialsError()
        if code == 404:
            return ModelUnavailableError()
        return GeminiError(f"The Gemini API returned an error (HTTP {code}). Please try again.")

    name = type(exc).__name__
    if "Connect" in name or "Timeout" in name or isinstance(exc, (ConnectionError, TimeoutError)):
        return GeminiNetworkError()
    return GeminiError()

