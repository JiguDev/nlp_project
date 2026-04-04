"""Text normalization helpers."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode text and collapse repeated whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_text(text: str) -> str:
    """Apply a lightweight cleaning pass suitable for multilingual corpora."""
    text = normalize_text(text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


def normalize_question(text: str) -> str:
    """Normalize a user question before retrieval or tokenization."""
    text = clean_text(text)
    if not text.endswith("?"):
        text = f"{text}?"
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into coarse sentences using punctuation heuristics."""
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?।])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def join_non_empty(parts: Iterable[str], separator: str = " ") -> str:
    """Join a sequence of strings while dropping empty fragments."""
    return separator.join(part.strip() for part in parts if part and part.strip())
