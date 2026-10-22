"""
Document-wide summarization service for IntelliDocs.
Processes whole documents across pages without relying on narrow top-k retrieval.
Supports single-stage synthesis for small/medium documents (e.g. 1-12 pages)
and hierarchical multi-stage synthesis for large documents (>12 pages).
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from django.conf import settings
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from documents.models import Document, DocumentChunk
from .errors import GeminiError, translate_exception
from .vector_store import SearchResult

logger = logging.getLogger(__name__)

MAX_SECTION_CHARS = 12000

SUMMARIZER_SYSTEM_INSTRUCTION = (
    "You are an expert Document Intelligence Analyst.\n"
    "Your objective is to provide a comprehensive, structured, factual document analysis "
    "based EXCLUSIVELY on the provided document pages.\n"
    "STRICT RULES:\n"
    "1. Rely ONLY on the facts explicitly stated in the evidence. Never invent topics, specifications, or diagram contents.\n"
    "2. For diagrams/figures: If the extracted text mentions a diagram (e.g. 'Block diagram of Microcomputer') "
    "but the visual graphic is not readable from text, state clearly that a diagram is mentioned on that page "
    "but its visual details cannot be interpreted from the text pipeline.\n"
    "3. Preserve all technical numbers, bit widths, units, and definitions faithfully.\n"
    "4. Cite every claim using the snippet identifier in the format [S<number>] (e.g. [S1], [S3]). "
    "List all used snippet numbers in used_snippets.\n"
    "5. Structure the summary with clear markdown sections: Document Overview, Topics & Subtopics, "
    "Core Definitions, Technical Specifications, Comparisons & Classifications, Diagrams & Figures, "
    "Exam-Relevant Revision Points, and Page Coverage."
)


class _SummaryModelResponse(BaseModel):
    overview: str = ""
    topics_and_subtopics: str = ""
    definitions: str = ""
    specifications: str = ""
    comparisons: str = ""
    diagrams_and_figures: str = ""
    exam_revision_points: str = ""
    covered_pages: List[int] = Field(default_factory=list)
    unprocessed_notes: str = ""
    used_snippets: List[int] = Field(default_factory=list)


class DocumentSummarizer:
    """Produces comprehensive, page-preserving summaries across an entire document."""

    def __init__(self, client: Optional[genai.Client] = None, model: Optional[str] = None):
        key = getattr(settings, "GEMINI_API_KEY", "").strip()
        self.model = model or getattr(settings, "GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
        self._client = client or (genai.Client(api_key=key) if key else None)

    def summarize(self, document_id: int, query_focus: str = "") -> Dict[str, Any]:
        """
        Executes document-wide analysis for a specific ready document.
        Returns:
            Dict containing 'answer', 'grounded', and 'sources'.
        """
        try:
            doc = Document.objects.get(id=document_id, status=Document.STATUS_READY)
        except Document.DoesNotExist:
            return {
                "answer": "The selected document is not ready or does not exist.",
                "grounded": False,
                "sources": [],
            }

        chunks = list(doc.chunks.order_by("chunk_index"))
        if not chunks:
            return {
                "answer": "The document contains no extracted text chunks to analyze.",
                "grounded": False,
                "sources": [],
            }

        # Build page-grouped snippets
        pages_dict: Dict[Optional[int], List[DocumentChunk]] = {}
        for c in chunks:
            pages_dict.setdefault(c.page_number, []).append(c)

        sorted_pages = sorted(pages_dict.keys(), key=lambda p: (p is None, p or 0))

        # Check total character length
        total_chars = sum(len(c.content) for c in chunks)

        if total_chars <= MAX_SECTION_CHARS or len(sorted_pages) <= 12:
            return self._single_pass_summary(doc, sorted_pages, pages_dict, query_focus)
        else:
            return self._hierarchical_summary(doc, sorted_pages, pages_dict, query_focus)

    def _single_pass_summary(
        self,
        doc: Document,
        sorted_pages: List[Optional[int]],
        pages_dict: Dict[Optional[int], List[DocumentChunk]],
        query_focus: str,
    ) -> Dict[str, Any]:
        evidence_blocks = []
        snippet_map: Dict[int, Dict[str, Any]] = {}
        s_idx = 1

        for p_num in sorted_pages:
            p_chunks = pages_dict[p_num]
            merged_text = "\n".join(c.content for c in p_chunks)
            page_label = f"Page {p_num}" if p_num else "Text Content"
            
            # Sanitize any accidental evidence tags inside text
            safe_text = re.sub(r"(?i)</?\s*evidence", "[evidence", merged_text)
            evidence_blocks.append(
                f'<evidence id="S{s_idx}" document="{doc.title}" page="{page_label}">\n{safe_text}\n</evidence>'
            )
            snippet_map[s_idx] = {
                "snippet_index": s_idx,
                "document_id": doc.id,
                "document_title": doc.title,
                "page_number": p_num,
                "similarity_score": 1.0,
                "excerpt": safe_text[:300] + ("..." if len(safe_text) > 300 else ""),
            }
            s_idx += 1

        combined_evidence = "\n\n".join(evidence_blocks)
        focus_prompt = f"User Request / Focus: {query_focus}\n" if query_focus else "User Request: Comprehensive Document Analysis\n"
        contents = (
            f"{combined_evidence}\n\n"
            f"{focus_prompt}"
            f"Analyze all {len(sorted_pages)} pages shown above. Return the structured JSON analysis."
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SUMMARIZER_SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=_SummaryModelResponse,
                ),
            )
            raw = response.text or ""
            data = json.loads(raw)
            parsed = _SummaryModelResponse(**data)
        except Exception as exc:
            logger.warning("Document summary generation failed: %s", type(exc).__name__)
            raise translate_exception(exc) from None

        return self._format_summary_answer(doc, parsed, snippet_map, sorted_pages)

    def _hierarchical_summary(
        self,
        doc: Document,
        sorted_pages: List[Optional[int]],
        pages_dict: Dict[Optional[int], List[DocumentChunk]],
        query_focus: str,
    ) -> Dict[str, Any]:
        # For large documents: group pages in batches of 6-8 pages, summarize sections, then synthesize
        batch_size = 6
        section_summaries = []
        all_snippet_map: Dict[int, Dict[str, Any]] = {}
        global_s_idx = 1

        for i in range(0, len(sorted_pages), batch_size):
            batch_pages = sorted_pages[i : i + batch_size]
            batch_blocks = []
            for p_num in batch_pages:
                p_text = "\n".join(c.content for c in pages_dict[p_num])
                page_label = f"Page {p_num}" if p_num else "Text Content"
                safe_text = re.sub(r"(?i)</?\s*evidence", "[evidence", p_text)
                batch_blocks.append(f'<evidence id="S{global_s_idx}" page="{page_label}">\n{safe_text}\n</evidence>')
                all_snippet_map[global_s_idx] = {
                    "snippet_index": global_s_idx,
                    "document_id": doc.id,
                    "document_title": doc.title,
                    "page_number": p_num,
                    "similarity_score": 1.0,
                    "excerpt": safe_text[:300] + ("..." if len(safe_text) > 300 else ""),
                }
                global_s_idx += 1

            batch_content = "\n\n".join(batch_blocks)
            sub_resp = self._client.models.generate_content(
                model=self.model,
                contents=f"{batch_content}\n\nSummarize pages {batch_pages[0]} to {batch_pages[-1]} preserving topic titles, definitions, specifications and citing [S<num>].",
                config=types.GenerateContentConfig(
                    system_instruction=SUMMARIZER_SYSTEM_INSTRUCTION,
                    temperature=0.1,
                ),
            )
            section_summaries.append(sub_resp.text or "")

        # Final synthesis of section summaries
        combined_sections = "\n\n---\n\n".join(section_summaries)
        synth_resp = self._client.models.generate_content(
            model=self.model,
            contents=(
                f"Section Summaries:\n{combined_sections}\n\n"
                f"User Request: {query_focus}\n"
                "Synthesize a coherent, final document analysis covering all sections. Output structured JSON."
            ),
            config=types.GenerateContentConfig(
                system_instruction=SUMMARIZER_SYSTEM_INSTRUCTION,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=_SummaryModelResponse,
            ),
        )
        data = json.loads(synth_resp.text or "{}")
        parsed = _SummaryModelResponse(**data)
        return self._format_summary_answer(doc, parsed, all_snippet_map, sorted_pages)

    def _format_summary_answer(
        self,
        doc: Document,
        parsed: _SummaryModelResponse,
        snippet_map: Dict[int, Dict[str, Any]],
        sorted_pages: List[Optional[int]],
    ) -> Dict[str, Any]:
        sections = []

        if parsed.overview:
            sections.append(f"### Document Overview\n{parsed.overview}")
        if parsed.topics_and_subtopics:
            sections.append(f"### Topics & Subtopics\n{parsed.topics_and_subtopics}")
        if parsed.definitions:
            sections.append(f"### Core Definitions\n{parsed.definitions}")
        if parsed.specifications:
            sections.append(f"### Technical Specifications & Values\n{parsed.specifications}")
        if parsed.comparisons:
            sections.append(f"### Comparisons & Classifications\n{parsed.comparisons}")
        if parsed.diagrams_and_figures:
            sections.append(f"### Diagrams, Figures & Tables\n{parsed.diagrams_and_figures}")
        if parsed.exam_revision_points:
            sections.append(f"### Exam-Relevant & Revision Points\n{parsed.exam_revision_points}")

        # Page Coverage
        pages_str = ", ".join(str(p) for p in sorted_pages if p is not None)
        coverage_note = f"**Document Coverage:** Analyzed all {len(sorted_pages)} page(s) ({pages_str if pages_str else 'full text'}) of `{doc.title}`."
        if parsed.unprocessed_notes:
            coverage_note += f"\n*Notes:* {parsed.unprocessed_notes}"
        sections.append(coverage_note)

        combined_text = "\n\n".join(sections)

        # Replace citation markers [S1], [S2], [S1, S3] with readable application metadata
        def replace_markers(text: str) -> str:
            def repl(m):
                content = m.group(1)
                nums = [int(n) for n in re.findall(r"\d+", content)]
                pages = []
                for n in nums:
                    if n in snippet_map:
                        p = snippet_map[n]["page_number"]
                        pages.append(f"Page {p}" if p else "TXT")
                if pages:
                    unique_pages = sorted(list(dict.fromkeys(pages)))
                    return f"[Source: {doc.title} — {', '.join(unique_pages)}]"
                return ""
            return re.sub(r"\[S([0-9\s,;]+)\]", repl, text)

        formatted_answer = replace_markers(combined_text)

        # Collect sources
        used_indices = set(parsed.used_snippets)
        for num_str in re.findall(r"\d+", "".join(re.findall(r"\[S([0-9\s,;]+)\]", combined_text))):
            used_indices.add(int(num_str))

        if not used_indices:
            # If model didn't tag specific snippets, all pages in snippet_map are evidence
            used_indices = set(snippet_map.keys())

        sources = [snippet_map[i] for i in sorted(used_indices) if i in snippet_map]

        return {
            "answer": formatted_answer,
            "grounded": True,
            "sources": sources,
        }

