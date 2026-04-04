"""Data loading and corpus preparation utilities."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from utils.io import iter_jsonl, list_files, write_jsonl, write_text
from utils.lang import add_language_token, detect_language, normalize_language_code
from utils.text import clean_text


def load_plain_text_corpus(path: str | Path, language: str = "en") -> List[dict]:
    """Load a text corpus where each non-empty line is a sample."""
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = clean_text(line)
            if text:
                detected_language = detect_language(text) if language in {"auto", "mixed"} else language
                records.append({"language": detected_language, "text": text})
    return records


def load_jsonl_corpus(path: str | Path, text_field: str = "text", language_field: str = "language") -> List[dict]:
    """Load a JSONL corpus with language and text fields."""
    records: List[dict] = []
    for item in iter_jsonl(path):
        text = clean_text(str(item.get(text_field, "")))
        language = str(item.get(language_field, "en"))
        if text:
            records.append({"language": language, "text": text})
    return records


def load_samanantar_tsv(path: str | Path, source_language: str, target_language: str) -> List[dict]:
    """Load a bilingual Samanantar-style TSV file with aligned sentence pairs."""
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            source_text = clean_text(parts[0])
            target_text = clean_text(parts[1])
            if source_text and target_text:
                records.append(
                    {
                        "source_language": source_language,
                        "target_language": target_language,
                        "source_text": source_text,
                        "target_text": target_text,
                    }
                )
    return records


def load_government_qa(path: str | Path) -> List[dict]:
    """Load government-style question-answer examples."""
    records: List[dict] = []
    for item in iter_jsonl(path):
        question = clean_text(str(item.get("question", "")))
        answer = clean_text(str(item.get("answer", "")))
        language = str(item.get("language", "en"))
        if question and answer:
            records.append({"language": language, "question": question, "answer": answer})
    return records


def build_tokenizer_corpus(
    input_dir: str | Path,
    output_path: str | Path,
    supported_languages: list[str] | tuple[str, ...] | set[str] | None = None,
) -> Path:
    """Merge all raw text corpora into one plain-text file for tokenizer training."""
    input_root = Path(input_dir)
    texts: List[str] = []
    normalized_supported = None
    if supported_languages is not None:
        normalized_supported = {normalize_language_code(language) for language in supported_languages}
    for path in list_files(input_root, suffixes=[".txt", ".jsonl"]):
        if path.name == Path(output_path).name:
            continue
        if path.suffix.lower() == ".txt":
            texts.extend(clean_text(line) for line in path.read_text(encoding="utf-8").splitlines() if clean_text(line))
        elif path.suffix.lower() == ".jsonl":
            for item in iter_jsonl(path):
                if normalized_supported is not None:
                    language = normalize_language_code(str(item.get("language", "en")))
                    if language not in normalized_supported:
                        continue
                for key in ("text", "title", "question", "answer", "source_text", "target_text"):
                    value = item.get(key)
                    if isinstance(value, str):
                        normalized = clean_text(value)
                        if normalized:
                            texts.append(normalized)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_text(output_file, "\n".join(texts))
    return output_file


def build_pretraining_pairs(multilingual_records: Iterable[dict]) -> List[dict]:
    """Convert raw multilingual samples into autoencoding pairs."""
    pairs: List[dict] = []
    for record in multilingual_records:
        language = record.get("language", "en")
        text = clean_text(str(record.get("text", "")))
        if not text:
            continue
        tagged = add_language_token(text, language)
        pairs.append({"language": language, "source": tagged, "target": tagged})
    return pairs


def build_finetuning_pairs(qa_records: Iterable[dict]) -> List[dict]:
    """Convert Q&A data into seq2seq fine-tuning samples."""
    pairs: List[dict] = []
    for record in qa_records:
        language = record.get("language", "en")
        question = clean_text(str(record.get("question", "")))
        answer = clean_text(str(record.get("answer", "")))
        if not question or not answer:
            continue
        pairs.append(
            {
                "language": language,
                "source": add_language_token(question, language),
                "target": add_language_token(answer, language),
            }
        )
    return pairs


def save_pairs_jsonl(path: str | Path, pairs: Iterable[dict]) -> Path:
    """Save sequence pairs to JSONL format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, pairs)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare multilingual government chatbot data")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory for processed artifacts")
    return parser.parse_args()


def main() -> None:
    """Prepare tokenizer and fine-tuning corpora."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    multilingual_records = []
    multilingual_txt = data_dir / "dummy" / "multilingual_corpus.txt"
    if multilingual_txt.exists():
        multilingual_records.extend(load_plain_text_corpus(multilingual_txt, language="mixed"))

    gov_qa_path = data_dir / "dummy" / "qa_pairs.jsonl"
    qa_records = load_government_qa(gov_qa_path) if gov_qa_path.exists() else []

    pretraining_pairs = build_pretraining_pairs(multilingual_records)
    finetuning_pairs = build_finetuning_pairs(qa_records)

    save_pairs_jsonl(processed_dir / "pretrain_pairs.jsonl", pretraining_pairs)
    save_pairs_jsonl(processed_dir / "finetune_pairs.jsonl", finetuning_pairs)
    print(f"Saved {len(pretraining_pairs)} pretraining pairs and {len(finetuning_pairs)} fine-tuning pairs.")


if __name__ == "__main__":
    main()
