"""
Page-aware text chunking service for Document Intelligence.
Preserves page-level metadata and chunk index.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TextChunk:
    chunk_index: int
    page_number: Optional[int]
    content: str


class TextChunker:
    """Splits extracted document pages into overlapping text chunks while retaining page metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be strictly greater than chunk_overlap.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[TextChunk]:
        """
        Chunks page data preserving page numbers.
        
        Args:
            pages_data: List of dicts with 'page_number' and 'text'.
            
        Returns:
            List of TextChunk instances with sequential chunk_index.
        """
        all_chunks: List[TextChunk] = []
        global_index = 0

        for page in pages_data:
            page_num = page.get("page_number")
            text = (page.get("text") or "").strip()

            if not text:
                continue

            page_chunks = self._split_text_recursively(text)
            for chunk_str in page_chunks:
                clean_chunk = chunk_str.strip()
                if clean_chunk:
                    all_chunks.append(
                        TextChunk(
                            chunk_index=global_index,
                            page_number=page_num,
                            content=clean_chunk,
                        )
                    )
                    global_index += 1

        return all_chunks

    def _split_text_recursively(self, text: str) -> List[str]:
        """
        Splits text by separators (paragraphs, newlines, sentences, spaces)
        to stay within chunk_size while maintaining chunk_overlap.
        """
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        
        # Pick the best separator available
        sep = separators[-1]
        for s in separators:
            if s == "":
                sep = s
                break
            if s in text:
                sep = s
                break

        splits = text.split(sep) if sep != "" else list(text)

        current_doc = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece) + (len(sep) if sep else 0)
            if current_len + piece_len > self.chunk_size and current_doc:
                merged = sep.join(current_doc) if sep else "".join(current_doc)
                final_chunks.append(merged)
                
                # Create overlap buffer from the tail of current_doc
                overlap_doc = []
                overlap_len = 0
                for item in reversed(current_doc):
                    item_full_len = len(item) + (len(sep) if sep else 0)
                    if overlap_len + item_full_len <= self.chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += item_full_len
                    else:
                        break
                current_doc = overlap_doc
                current_len = overlap_len

            current_doc.append(piece)
            current_len += piece_len

        if current_doc:
            merged = sep.join(current_doc) if sep else "".join(current_doc)
            final_chunks.append(merged)

        # Ensure no individual chunk exceeds chunk_size
        refined_chunks: List[str] = []
        for c in final_chunks:
            if len(c) > self.chunk_size and len(separators) > 1:
                # Sub-split with remaining separators
                sub_splits = self._recursive_split(c, separators[1:])
                refined_chunks.extend(sub_splits)
            else:
                refined_chunks.append(c)

        return refined_chunks

