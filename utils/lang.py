"""Language detection and control-token helpers."""
from __future__ import annotations

from typing import Dict

try:
    from langdetect import detect, detect_langs  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback when dependency is missing
    detect = None
    detect_langs = None


SUPPORTED_LANGUAGE_TOKENS: Dict[str, str] = {
    "en": "<en>",
    "hi": "<hi>",
    "gu": "<gu>",
}


LANGUAGE_ALIASES: Dict[str, str] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "hi-in": "hi",
    "gu": "gu",
    "guj": "gu",
    "gujarati": "gu",
    "gu-in": "gu",
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
    return None


def normalize_language_code(language_code: str | None, default: str = "en") -> str:
    """Normalize language aliases to canonical project codes."""
    if not language_code:
        return default
    normalized = language_code.strip().lower().replace("_", "-")
    if normalized.startswith("en"):
        return "en"
    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]
    return default


def is_supported_language(language_code: str, supported_languages: list[str] | tuple[str, ...] | set[str]) -> bool:
    """Return whether a language code belongs to the supported language set."""
    code = normalize_language_code(language_code)
    normalized_supported = {normalize_language_code(item) for item in supported_languages}
    return code in normalized_supported


def detect_language(text: str, default: str = "en") -> str:
    """Detect the dominant language code for a query or document."""
    normalized_default = normalize_language_code(default)
    script_language = _script_based_language(text)
    if script_language is not None:
        return script_language
    if detect is None:
        return normalized_default
    try:
        detected = detect(text)
    except Exception:
        return normalized_default
    normalized_detected = normalize_language_code(detected, default=normalized_default)
    return normalized_detected if normalized_detected in SUPPORTED_LANGUAGE_TOKENS else normalized_default


def detect_language_with_confidence(text: str, default: str = "en") -> tuple[str, float]:
    """Return a language code and an approximate confidence score."""
    normalized_default = normalize_language_code(default)
    script_language = _script_based_language(text)
    if script_language is not None:
        return script_language, 0.99
    if detect_langs is None:
        return normalized_default, 0.5
    try:
        candidates = detect_langs(text)
    except Exception:
        return normalized_default, 0.5
    if not candidates:
        return normalized_default, 0.5
    best = candidates[0]
    language = normalize_language_code(best.lang, default=normalized_default)
    if language not in SUPPORTED_LANGUAGE_TOKENS:
        language = normalized_default
    return language, float(best.prob)
