import re
import uuid

from app.chunking.models import Chunk
from app.ingestion.models import Document

TARGET_CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_document(document: Document, *, document_id: str | None = None) -> list[Chunk]:
    """Split a parsed document into ordered, page-scoped text chunks."""
    resolved_document_id = document_id or str(uuid.uuid4())
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        page_chunks = _chunk_page_text(page.text)
        for text in page_chunks:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=resolved_document_id,
                    document_title=document.document_title,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                    char_count=len(text),
                )
            )
            chunk_index += 1

    return chunks


def _chunk_page_text(text: str) -> list[str]:
    """Chunk a single page without crossing into other pages."""
    units = _text_to_units(text)
    if not units:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for unit in units:
        while True:
            separator_length = _separator_length(current_parts)
            projected_length = current_length + separator_length + len(unit)

            if projected_length <= TARGET_CHUNK_SIZE:
                current_parts.append(unit)
                current_length = projected_length
                break

            if current_parts:
                chunk_text = _join_parts(current_parts)
                chunks.append(chunk_text)
                overlap_prefix = _overlap_prefix(chunk_text)
                if (
                    overlap_prefix
                    and len(overlap_prefix) + _separator_length([overlap_prefix]) + len(unit)
                    <= TARGET_CHUNK_SIZE
                ):
                    current_parts = [overlap_prefix]
                    current_length = len(overlap_prefix)
                else:
                    current_parts = []
                    current_length = 0
                continue

            first_chunk = unit[:TARGET_CHUNK_SIZE]
            chunks.append(first_chunk)
            overlap = _word_aware_suffix(first_chunk, CHUNK_OVERLAP)
            unit = unit[len(first_chunk) - len(overlap) :]
            if not unit:
                break

    if current_parts:
        chunks.append(_join_parts(current_parts))

    return chunks


def _text_to_units(text: str) -> list[str]:
    """Decompose page text into units that respect the chunk size limit."""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    units: list[str] = []
    for paragraph in _split_paragraphs(stripped_text):
        if len(paragraph) <= TARGET_CHUNK_SIZE:
            units.append(paragraph)
            continue

        for sentence in _split_sentences(paragraph):
            if len(sentence) <= TARGET_CHUNK_SIZE:
                units.append(sentence)
            else:
                units.extend(_split_by_characters(sentence))

    return units


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines, preserving non-empty paragraphs."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    return paragraphs or [text.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences when it exceeds the chunk size limit."""
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    return parts or [text.strip()]


def _split_by_characters(text: str) -> list[str]:
    """Split oversized sentences into fixed-size character segments with overlap."""
    if len(text) <= TARGET_CHUNK_SIZE:
        return [text]

    segments: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + TARGET_CHUNK_SIZE, len(text))
        segments.append(text[start:end])
        if end >= len(text):
            break
        next_start = end - len(_word_aware_suffix(text[:end], CHUNK_OVERLAP))
        start = max(next_start, start + 1)

    return segments


def _join_parts(parts: list[str]) -> str:
    """Join chunk parts using paragraph boundaries when possible."""
    return "\n\n".join(parts)


def _separator_length(parts: list[str]) -> int:
    """Return the separator length required before appending the next part."""
    return len("\n\n") if parts else 0


def _overlap_prefix(chunk_text: str) -> str:
    """Return the trailing overlap reused at the start of the next chunk."""
    return _word_aware_suffix(chunk_text, CHUNK_OVERLAP)


def _word_aware_suffix(text: str, target_size: int) -> str:
    """Return a trailing suffix near target_size that starts on a word boundary."""
    if target_size <= 0 or not text:
        return ""
    if len(text) <= target_size:
        return text

    start = _align_start_to_word_boundary(text, len(text) - target_size)
    suffix = text[start:].lstrip()
    if len(suffix) > target_size * 2:
        return text[-target_size:]
    return suffix


def _align_start_to_word_boundary(text: str, start: int) -> int:
    """Move a start index backward so the suffix does not begin mid-word."""
    if start <= 0:
        return 0

    if text[start].isspace():
        while start < len(text) and text[start].isspace():
            start += 1
        return start

    if text[start - 1].isspace():
        return start

    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return start
