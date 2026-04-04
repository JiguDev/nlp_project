"""Language detection and control-token helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from langdetect import detect, detect_langs
except Exception:  # pragma: no cover - fallback when dependency is missing
    detect = None
    detect_langs = None


SUPPORTED_LANGUAGE_TOKENS: Dict[str, str] = {
    "en": "<en>",
    "hi": "<hi>",
    "gu": "<gu>",
    "ta": "<ta>",
    "bn": "<bn>",
    "mr": "<mr>",
    "te": "<te>",
    "kn": "<kn>",
    "ml": "<ml>",
    "pa": "<pa>",
}


def add_language_token(text: str, language_code: str) -> str:
    """Prefix text with a language token used by the shared tokenizer."""
    token = SUPPORTED_LANGUAGE_TOKENS.get(language_code, f"<{language_code}>")
    return f"{token} {text.strip()}"


def strip_language_token(text: str) -> str:
    """Remove a leading language token if present."""
    stripped = text.strip()
    if stripped.startswith("<") and ">" in stripped:
        return stripped.split(" ", 1)[1] if " " in stripped else ""
    return stripped


def _script_based_language(text: str) -> str | None:
    """Infer a language from Unicode script ranges."""
    for character in text:
        codepoint = ord(character)
        if 0x0900 <= codepoint <= 0x097F:
            return "hi"
        if 0x0A80 <= codepoint <= 0x0AFF:
            return "gu"
        if 0x0B80 <= codepoint <= 0x0BFF:
            return "ta"
        if 0x0980 <= codepoint <= 0x09FF:
            return "bn"
        if 0x0C00 <= codepoint <= 0x0C7F:
            return "te"
        if 0x0C80 <= codepoint <= 0x0CFF:
            return "kn"
        if 0x0D00 <= codepoint <= 0x0D7F:
            return "ml"
        if 0x0A00 <= codepoint <= 0x0A7F:
            return "pa"
    return None


def detect_language(text: str, default: str = "en") -> str:
    """Detect the dominant language code for a query or document."""
    script_language = _script_based_language(text)
    if script_language is not None:
        return script_language
    if detect is None:
        return default
    try:
        detected = detect(text)
    except Exception:
        return default
    if detected in SUPPORTED_LANGUAGE_TOKENS:
        return detected
    if detected.startswith("en"):
        return "en"
    return default


def detect_language_with_confidence(text: str, default: str = "en") -> tuple[str, float]:
    """Return a language code and an approximate confidence score."""
    script_language = _script_based_language(text)
    if script_language is not None:
        return script_language, 0.99
    if detect_langs is None:
        return default, 0.5
    try:
        candidates = detect_langs(text)
    except Exception:
        return default, 0.5
    if not candidates:
        return default, 0.5
    best = candidates[0]
    language = best.lang
    if language not in SUPPORTED_LANGUAGE_TOKENS and not language.startswith("en"):
        language = default
    return language if not language.startswith("en") else "en", float(best.prob)
