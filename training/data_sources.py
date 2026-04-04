"""Production-grade ingestion utilities for multilingual and government corpora."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Iterator, List, Sequence

from utils.io import iter_jsonl
from utils.lang import detect_language, is_supported_language, normalize_language_code
from utils.text import clean_text

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    requests = None
    BeautifulSoup = None


@dataclass
class RawRecord:
    """Unified raw text record."""

    text: str
    language: str
    source: str
    source_id: str = ""
    title: str = ""


@dataclass
class QARecord:
    """Unified QA record for domain fine-tuning and evaluation."""

    question: str
    answer: str
    language: str
    source: str
    gold_doc_id: str = ""


def _normalize_record(text: str, language: str, source: str, source_id: str = "", title: str = "") -> RawRecord | None:
    normalized_text = clean_text(text)
    if not normalized_text:
        return None
    normalized_language = language if language and language != "auto" else detect_language(normalized_text)
    return RawRecord(text=normalized_text, language=normalized_language, source=source, source_id=source_id, title=clean_text(title))


def load_indic_jsonl(paths: Sequence[Path], text_fields: Sequence[str] = ("text",), language_field: str = "language") -> List[RawRecord]:
    """Load Indic-style multilingual JSONL corpora."""
    output: List[RawRecord] = []
    for path in paths:
        for item in iter_jsonl(path):
            language = str(item.get(language_field, "auto"))
            for field in text_fields:
                value = item.get(field)
                if isinstance(value, str):
                    record = _normalize_record(
                        text=value,
                        language=language,
                        source=f"indic:{path.name}",
                        source_id=str(item.get("id", "")),
                        title=str(item.get("title", "")),
                    )
                    if record is not None:
                        output.append(record)
    return output


def load_indic_plain_text(path: Path, language: str, max_lines: int | None = None) -> List[RawRecord]:
    """Load line-based plain text corpora such as IndicCorp language shards."""
    output: List[RawRecord] = []
    normalized_language = normalize_language_code(language)
    if not path.exists() or not path.is_file():
        return output
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            text = clean_text(line)
            if not text:
                continue
            if max_lines is not None and len(output) >= max_lines:
                break
            output.append(
                RawRecord(
                    text=text,
                    language=normalized_language,
                    source=f"indiccorp:{path.name}",
                    source_id=f"{path.stem}:{line_index}",
                    title="",
                )
            )
    return output


def load_samanantar_tsv(paths: Sequence[Path], src_lang: str, tgt_lang: str) -> List[RawRecord]:
    """Load Samanantar parallel TSV and flatten into multilingual text records."""
    output: List[RawRecord] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                src, tgt = clean_text(parts[0]), clean_text(parts[1])
                if src:
                    output.append(RawRecord(text=src, language=src_lang, source=f"samanantar:{path.name}", source_id=f"{path.stem}:{line_index}:src"))
                if tgt:
                    output.append(RawRecord(text=tgt, language=tgt_lang, source=f"samanantar:{path.name}", source_id=f"{path.stem}:{line_index}:tgt"))
    return output


def load_government_jsonl(paths: Sequence[Path]) -> List[RawRecord]:
    """Load government/legal JSONL files into unified records."""
    output: List[RawRecord] = []
    for path in paths:
        for item in iter_jsonl(path):
            text = str(item.get("text", ""))
            title = str(item.get("title", ""))
            language = str(item.get("language", "auto"))
            record = _normalize_record(
                text=text,
                language=language,
                source=f"gov:{path.name}",
                source_id=str(item.get("id", "")),
                title=title,
            )
            if record is not None:
                output.append(record)
    return output


def load_government_csv(paths: Sequence[Path], text_column: str = "text", language_column: str = "language", title_column: str = "title") -> List[RawRecord]:
    """Load government/legal CSV files into unified records."""
    output: List[RawRecord] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                text = str(row.get(text_column, ""))
                language = str(row.get(language_column, "auto"))
                title = str(row.get(title_column, ""))
                record = _normalize_record(
                    text=text,
                    language=language,
                    source=f"gov:{path.name}",
                    source_id=f"{path.stem}:{row_index}",
                    title=title,
                )
                if record is not None:
                    output.append(record)
    return output


def load_government_qa_jsonl(paths: Sequence[Path]) -> List[QARecord]:
    """Load government QA supervision for fine-tuning and evaluation."""
    output: List[QARecord] = []
    for path in paths:
        for item in iter_jsonl(path):
            question = clean_text(str(item.get("question", "")))
            answer = clean_text(str(item.get("answer", "")))
            language = str(item.get("language", "auto"))
            if not question or not answer:
                continue
            if language == "auto":
                language = detect_language(question)
            output.append(
                QARecord(
                    question=question,
                    answer=answer,
                    language=language,
                    source=f"qa:{path.name}",
                    gold_doc_id=str(item.get("gold_doc_id", "")),
                )
            )
    return output


def scrape_government_urls(urls: Sequence[str], timeout: int = 15) -> List[RawRecord]:
    """Scrape official pages and convert them into raw records.

    This is intentionally conservative: it extracts visible paragraph text only.
    """
    if requests is None or BeautifulSoup is None:
        raise ImportError("requests and beautifulsoup4 are required for URL scraping")
    output: List[RawRecord] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GovChatbotResearchBot/1.0)"}
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
        except Exception:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        title = clean_text(soup.title.get_text(" ")) if soup.title else ""
        paragraphs = [clean_text(node.get_text(" ")) for node in soup.find_all("p")]
        text = clean_text(" ".join(paragraph for paragraph in paragraphs if paragraph))
        if not text:
            continue
        output.append(
            RawRecord(
                text=text,
                language=detect_language(text),
                source="gov:web",
                source_id=url,
                title=title,
            )
        )
    return output


def deduplicate_records(records: Iterable[RawRecord]) -> List[RawRecord]:
    """Deduplicate records using (language, text) fingerprints."""
    seen: set[tuple[str, str]] = set()
    output: List[RawRecord] = []
    for record in records:
        key = (record.language, record.text)
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def cap_records_per_language(records: Iterable[RawRecord], max_records_per_language: int | None = None) -> List[RawRecord]:
    """Limit each language to a fixed number of records while preserving input order."""
    if max_records_per_language is None or max_records_per_language <= 0:
        return list(records)
    counts = defaultdict(int)
    output: List[RawRecord] = []
    for record in records:
        language = normalize_language_code(record.language)
        if counts[language] >= max_records_per_language:
            continue
        counts[language] += 1
        output.append(record)
    return output


def filter_records_by_languages(records: Iterable[RawRecord], supported_languages: Sequence[str]) -> List[RawRecord]:
    """Keep only records whose language belongs to the supported set."""
    normalized_supported = [normalize_language_code(language) for language in supported_languages]
    output: List[RawRecord] = []
    for record in records:
        language = normalize_language_code(record.language)
        if not is_supported_language(language, normalized_supported):
            continue
        output.append(
            RawRecord(
                text=record.text,
                language=language,
                source=record.source,
                source_id=record.source_id,
                title=record.title,
            )
        )
    return output


def filter_qa_by_languages(records: Iterable[QARecord], supported_languages: Sequence[str]) -> List[QARecord]:
    """Keep only QA entries from supported languages."""
    normalized_supported = [normalize_language_code(language) for language in supported_languages]
    output: List[QARecord] = []
    for record in records:
        language = normalize_language_code(record.language)
        if not is_supported_language(language, normalized_supported):
            continue
        output.append(
            QARecord(
                question=record.question,
                answer=record.answer,
                language=language,
                source=record.source,
                gold_doc_id=record.gold_doc_id,
            )
        )
    return output


def expand_globs(globs: Sequence[str]) -> List[Path]:
    """Expand glob patterns into sorted files."""
    files: List[Path] = []
    for pattern in globs:
        files.extend(Path().glob(pattern))
    return sorted([path for path in files if path.is_file()])
