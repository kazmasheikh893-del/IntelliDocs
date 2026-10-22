"""
Query analysis and intent classification for Document Intelligence.
Detects document-wide summary requests, definitions, comparisons,
page-specific constraints, and standard factual questions.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class QueryIntent(str, Enum):
    DOCUMENT_SUMMARY = "document_summary"
    PAGE_SPECIFIC = "page_specific"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    FACTUAL = "factual"


@dataclass
class AnalyzedQuery:
    raw_query: str
    cleaned_query: str
    intent: QueryIntent
    target_page: Optional[int] = None
    is_multi_part: bool = False
    wants_simple_explanation: bool = False


_SUMMARY_PATTERNS = [
    r"\banalyze\s+(?:all|the|entire|each|every)\s+(?:\w+\s+)?pages?\b",
    r"\bsummar(?:y|ize)\b.*?\b(?:document|pdf|pages?)\b",
    r"\b(?:document|pdf)\b.*?\bsummar(?:y|ize)\b",
    r"\b(?:entire|whole|all)\s+document\s+summar(?:y|ize)\b",
    r"\ball\s+\d+\s+pages?\b",
    r"\boverview\s+of\s+(?:the\s+)?(?:entire|whole|all)?\s*document\b",
    r"\btopics?\s+and\s+subtopics?\b",
    r"\bidentify\s+(?:the\s+)?topics?\b",
    r"\bexam[\s-]relevant\s+material\b",
    r"\bstudy\s+roadmap\b",
    r"\bexam\s+notes\b",
    r"\bsyllabus\b",
    r"\bdocument-level\s+summary\b",
    r"\bwhat\s+(?:is|are)\s+in\s+this\s+document\b",
    r"\bwhat\s+does\s+this\s+document\s+cover\b",
]

_DEFINITION_PATTERNS = [
    r"^\s*defi(?:ne|nition\s+of)\b",
    r"\bwhat\s+is\s+(?:the\s+)?definition\s+of\b",
    r"\bwhat\s+is\s+(?:a|an|the)\s+([a-zA-Z0-9_\- ]+)\b",
    r"\bwhat\s+(?:do\s+you\s+mean\s+by|is\s+meant\s+by)\b",
    r"\bmeaning\s+of\b",
    r"\bdefine\s+and\s+explain\b",
]

_COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\s+(?:between|of|table)?\b",
    r"\bdiffer(?:ence|ences)?\s+between\b",
    r"\bversus\b|\bvs\.?\b",
    r"\bcontrast\b",
]

_PAGE_PATTERNS = [
    r"\b(?:on|from|in|at)\s+page\s+(\d+)\b",
    r"\bpage\s+(\d+)\s*(?:summary|content|details)?\b",
]


class QueryAnalyzer:
    """Analyzes a user query to identify intent, page scope, and special output requirements."""

    @classmethod
    def analyze(cls, query: str) -> AnalyzedQuery:
        clean = (query or "").strip()
        lower = clean.lower()

        # 1. Page-specific detection
        target_page = None
        for pat in _PAGE_PATTERNS:
            m = re.search(pat, lower)
            if m:
                try:
                    target_page = int(m.group(1))
                    break
                except (ValueError, IndexError):
                    pass

        # 2. Simple words flag
        wants_simple = bool(
            re.search(r"\b(?:in\s+)?simple\s+(?:words|terms|language|english)\b|\bsimply\b|\bfor\s+beginners?\b", lower)
        )

        # 3. Multi-part detection
        is_multi_part = bool(
            ("?" in clean and clean.count("?") > 1)
            or re.search(r"\b(?:also|and\s+also|as\s+well\s+as|both\s+.*\s+and|part\s*1|part\s*2)\b", lower)
        )

        # 4. Intent classification
        # Document summary check
        is_doc_summary = any(re.search(pat, lower) for pat in _SUMMARY_PATTERNS)
        if is_doc_summary and target_page is None:
            return AnalyzedQuery(
                raw_query=query,
                cleaned_query=clean,
                intent=QueryIntent.DOCUMENT_SUMMARY,
                target_page=None,
                is_multi_part=is_multi_part,
                wants_simple_explanation=wants_simple,
            )

        if target_page is not None and any(re.search(pat, lower) for pat in [r"\bsummar(?:y|ize)\b", r"\boverview\b", r"\bdetails?\b"]):
            return AnalyzedQuery(
                raw_query=query,
                cleaned_query=clean,
                intent=QueryIntent.PAGE_SPECIFIC,
                target_page=target_page,
                is_multi_part=is_multi_part,
                wants_simple_explanation=wants_simple,
            )

        # Comparison check
        if any(re.search(pat, lower) for pat in _COMPARISON_PATTERNS):
            return AnalyzedQuery(
                raw_query=query,
                cleaned_query=clean,
                intent=QueryIntent.COMPARISON,
                target_page=target_page,
                is_multi_part=is_multi_part,
                wants_simple_explanation=wants_simple,
            )

        # Definition check
        if any(re.search(pat, lower) for pat in _DEFINITION_PATTERNS):
            return AnalyzedQuery(
                raw_query=query,
                cleaned_query=clean,
                intent=QueryIntent.DEFINITION,
                target_page=target_page,
                is_multi_part=is_multi_part,
                wants_simple_explanation=wants_simple,
            )

        # Page-specific fallback
        if target_page is not None:
            return AnalyzedQuery(
                raw_query=query,
                cleaned_query=clean,
                intent=QueryIntent.PAGE_SPECIFIC,
                target_page=target_page,
                is_multi_part=is_multi_part,
                wants_simple_explanation=wants_simple,
            )

        # Default factual
        return AnalyzedQuery(
            raw_query=query,
            cleaned_query=clean,
            intent=QueryIntent.FACTUAL,
            target_page=None,
            is_multi_part=is_multi_part,
            wants_simple_explanation=wants_simple,
        )
