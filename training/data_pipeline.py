"""Data loading and corpus preparation utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from utils.io import iter_jsonl, write_jsonl
from utils.lang import add_language_token, detect_language
from utils.text import clean_text

def load_plain_text_corpus(path: str | Path, language: str = "en", limit: int = 26000) -> List[dict]:
    """Load a text corpus with limit control and progress logs."""
    records: List[dict] = []

    print(f"\nLoading: {path}")

    with Path(path).open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):

            if i >= limit:
                break

            text = clean_text(line)

            if text:
                detected_language = detect_language(text) if language in {"auto", "mixed"} else language

                records.append({
                    "language": detected_language,
                    "text": text
                })

            # Progress log
            if i % 500 == 0:
                print(f"{Path(path).name}: {i} lines processed")

    print(f"Finished {path} → {len(records)} records\n")
    return records


# =========================
def load_government_qa(path: str | Path) -> List[dict]:
    """Load government-style question-answer examples."""
    records: List[dict] = []

    for item in iter_jsonl(path):
        question = clean_text(str(item.get("question", "")))
        answer = clean_text(str(item.get("answer", "")))
        language = str(item.get("language", "en"))

        if question and answer:
            records.append({
                "language": language,
                "question": question,
                "answer": answer
            })

    return records


# =========================
def build_pretraining_pairs(multilingual_records: Iterable[dict]) -> List[dict]:
    """Convert raw multilingual samples into autoencoding pairs."""
    pairs: List[dict] = []

    for record in multilingual_records:
        language = record.get("language", "en")
        text = clean_text(str(record.get("text", "")))

        if not text:
            continue

        tagged = add_language_token(text, language)

        pairs.append({
            "language": language,
            "source": tagged,
            "target": tagged
        })

    return pairs


# =========================
def build_finetuning_pairs(qa_records: Iterable[dict]) -> List[dict]:
    """Convert Q&A data into seq2seq fine-tuning samples."""
    pairs: List[dict] = []

    for record in qa_records:
        language = record.get("language", "en")
        question = clean_text(str(record.get("question", "")))
        answer = clean_text(str(record.get("answer", "")))

        if not question or not answer:
            continue

        pairs.append({
            "language": language,
            "source": add_language_token(question, language),
            "target": add_language_token(answer, language),
        })

    return pairs


# =========================
def save_pairs_jsonl(path: str | Path, pairs: Iterable[dict]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, pairs)
    return output_path


# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multilingual government chatbot data")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    return parser.parse_args()


# =========================
def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    multilingual_records = []

    print("\n==============================")
    print("Loading IndicCorp datasets...")
    print("==============================\n")

    # DATA FILES
    en_path = data_dir / "multilingual" / "indiccorp" / "en.txt"
    hi_path = data_dir / "multilingual" / "indiccorp" / "hi-1.txt"
    gu_path = data_dir / "multilingual" / "indiccorp" / "gu.txt"

    LIMIT = 26000 # Adjust as needed based on your resources and requirements

    if en_path.exists():
        multilingual_records.extend(load_plain_text_corpus(en_path, "en", LIMIT))

    if hi_path.exists():
        multilingual_records.extend(load_plain_text_corpus(hi_path, "hi", LIMIT))

    if gu_path.exists():
        multilingual_records.extend(load_plain_text_corpus(gu_path, "gu", LIMIT))

    print(f"\nTotal multilingual samples: {len(multilingual_records)}\n")

    # =========================
    # QA DATA (optional)
    qa_records = []

    qa_path = data_dir / "dummy" / "qa_pairs.jsonl"
    if qa_path.exists():
        qa_records.extend(load_government_qa(qa_path))

    print(f"QA samples: {len(qa_records)}\n")

    # =========================
    print("Building training pairs...\n")

    pretraining_pairs = build_pretraining_pairs(multilingual_records)
    finetuning_pairs = build_finetuning_pairs(qa_records)

    save_pairs_jsonl(processed_dir / "pretrain_pairs.jsonl", pretraining_pairs)
    save_pairs_jsonl(processed_dir / "finetune_pairs.jsonl", finetuning_pairs)

    print("\n==============================")
    print("✅ DONE")
    print("==============================")
    print(f"Pretraining pairs: {len(pretraining_pairs)}")
    print(f"Finetuning pairs: {len(finetuning_pairs)}\n")


# =========================
if __name__ == "__main__":
    main()