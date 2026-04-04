"""Document chunking utilities for retrieval augmented generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from utils.text import clean_text, split_sentences


@dataclass
class DocumentChunk:
    """A chunk of a larger government document."""

    chunk_id: str
    document_id: str
    language: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word chunks."""
    words = clean_text(text).split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_document(document: dict, chunk_size: int = 80, overlap: int = 20) -> List[DocumentChunk]:
    """Chunk a single document record into searchable text segments."""
    chunks: List[DocumentChunk] = []
    base_text = str(document.get("text", ""))
    for index, chunk in enumerate(chunk_text(base_text, chunk_size=chunk_size, overlap=overlap)):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.get('id', 'doc')}_{index}",
                document_id=str(document.get("id", "doc")),
                language=str(document.get("language", "en")),
                title=str(document.get("title", "")),
                text=chunk,
                metadata={"source": document.get("source", "government")},
            )
        )
    return chunks


def build_chunks(documents: Iterable[dict], chunk_size: int = 80, overlap: int = 20) -> List[DocumentChunk]:
    """Build chunks for an entire corpus of documents."""
    output: List[DocumentChunk] = []
    for document in documents:
        output.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return output
