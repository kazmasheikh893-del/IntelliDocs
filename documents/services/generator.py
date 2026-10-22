"""
Grounded Answer Generation Service using official Google Gemini API.
Enforces strict grounding, citation construction with page numbers,
and deterministic fallback when evidence is insufficient.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from django.conf import settings
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .embeddings import GeminiEmbeddingService, MissingApiKeyError
from .vector_store import BaseVectorStore, ORMVectorStore, SearchResult


INSUFFICIENT_EVIDENCE_ANSWER = "The answer was not found in the provided documents."


@dataclass
class QAResult:
    question: str
    answer: str
    grounded: bool
    sources: List[Dict[str, Any]]


class GeminiQAService:
    """Answers user queries grounded strictly in retrieved document passages."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vector_store: Optional[BaseVectorStore] = None,
        similarity_threshold: float = 0.35,
        top_k: int = 4,
    ):
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", "")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise MissingApiKeyError(
                "GEMINI_API_KEY is not configured. Please add your real Gemini API key to .env"
            )
        self.model = model or getattr(settings, "GEMINI_GENERATION_MODEL", "gemini-2.0-flash")
        self.vector_store = vector_store or ORMVectorStore()
        self.embedding_service = GeminiEmbeddingService(api_key=self.api_key)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self._client = genai.Client(api_key=self.api_key)

    def answer_question(
        self,
        question: str,
        document_id: Optional[int] = None,
    ) -> QAResult:
        clean_q = (question or "").strip()
        if not clean_q:
            return QAResult(
                question="",
                answer="Please enter a valid question.",
                grounded=False,
                sources=[],
            )

        # 1. Embed query
        query_embedding = self.embedding_service.embed_text(clean_q)

        # 2. Retrieve relevant passages
        matches: List[SearchResult] = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            document_id=document_id,
            min_similarity=self.similarity_threshold,
        )

        # 3. Handle insufficient evidence early if no passages meet similarity threshold
        if not matches:
            return QAResult(
                question=clean_q,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                grounded=False,
                sources=[],
            )

        # 4. Construct grounded prompt
        context_snippets = []
        sources = []
        for idx, m in enumerate(matches, 1):
            page_info = f"Page {m.page_number}" if m.page_number else "Full Text"
            context_snippets.append(
                f"[Snippet #{idx} | Document: '{m.document_title}' | {page_info}]\n{m.content}"
            )
            sources.append({
                "snippet_index": idx,
                "document_id": m.document_id,
                "document_title": m.document_title,
                "page_number": m.page_number,
                "similarity_score": round(m.similarity_score, 3),
                "excerpt": m.content[:250] + ("..." if len(m.content) > 250 else ""),
            })

        combined_context = "\n\n".join(context_snippets)

        prompt = (
            f"You are a rigorous Document Intelligence Assistant. Your objective is to answer the user's question "
            f"grounded EXCLUSIVELY in the provided document excerpts below.\n\n"
            f"STRICT RULES:\n"
            f"1. Rely ONLY on the clear facts provided in the Context Snippets. Do NOT assume, extrapolate, or bring outside knowledge.\n"
            f"2. Every factual statement in your answer MUST cite its source using the snippet metadata in format: [Doc: <title>, Page <N>] or [Doc: <title>].\n"
            f"3. If the provided excerpts do not contain sufficient evidence to answer the question, respond with EXACTLY:\n"
            f"\"{INSUFFICIENT_EVIDENCE_ANSWER}\"\n"
            f"4. Do not invent citations or fabricate quotes.\n\n"
            f"CONTEXT SNIPPETS:\n"
            f"{combined_context}\n\n"
            f"USER QUESTION: {clean_q}\n\n"
            f"GROUNDED ANSWER:"
        )

        # 5. Call Gemini LLM
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Low temperature for factual grounding
                    top_p=0.95,
                ),
            )
            answer_text = (response.text or "").strip()

            # Determine whether the model indicated insufficient evidence
            is_insufficient = (
                INSUFFICIENT_EVIDENCE_ANSWER.lower() in answer_text.lower()
                or "not found in the provided documents" in answer_text.lower()
            )

            if is_insufficient:
                return QAResult(
                    question=clean_q,
                    answer=INSUFFICIENT_EVIDENCE_ANSWER,
                    grounded=False,
                    sources=sources,
                )

            return QAResult(
                question=clean_q,
                answer=answer_text,
                grounded=True,
                sources=sources,
            )

        except APIError as e:
            return QAResult(
                question=clean_q,
                answer=f"Gemini API Error: {e.message} (HTTP {e.code})",
                grounded=False,
                sources=[],
            )
        except Exception as e:
            return QAResult(
                question=clean_q,
                answer=f"An unexpected error occurred while generating the answer: {str(e)}",
                grounded=False,
                sources=[],
            )
