"""Ingest real multilingual and government datasets into unified project corpora."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from training.data_pipeline import build_finetuning_pairs, build_pretraining_pairs, save_pairs_jsonl
from training.data_sources import (
    QARecord,
    RawRecord,
    cap_records_per_language,
    deduplicate_records,
    expand_globs,
    filter_qa_by_languages,
    filter_records_by_languages,
    load_government_csv,
    load_government_jsonl,
    load_government_qa_jsonl,
    load_indic_jsonl,
    load_indic_plain_text,
    load_samanantar_tsv,
    scrape_government_urls,
)
from utils.config import load_config
from utils.io import write_jsonl, write_text
from utils.lang import normalize_language_code


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for scalable data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest real Indic/Samanantar/Government datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--scrape", action="store_true", help="Enable URL scraping for government pages")
    return parser.parse_args()


def _load_urls(path: Path) -> List[str]:
    if not path.exists():
        return []
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _records_to_jsonl(records: List[RawRecord]) -> List[dict]:
    return [
        {
            "text": record.text,
            "language": record.language,
            "source": record.source,
            "source_id": record.source_id,
            "title": record.title,
        }
        for record in records
    ]


def _qa_to_jsonl(records: List[QARecord]) -> List[dict]:
    return [
        {
            "question": record.question,
            "answer": record.answer,
            "language": record.language,
            "source": record.source,
            "gold_doc_id": record.gold_doc_id,
        }
        for record in records
    ]


def main() -> None:
    """Run ingestion and emit unified corpora for training and retrieval."""
    args = parse_args()
    config = load_config(args.config)

    ingestion = config.get("ingestion", {})
    paths = config.get("paths", {})
    output_raw_text = Path(paths.get("raw_corpus", "data/processed/raw_corpus.jsonl"))
    output_raw_qa = Path(paths.get("raw_qa", "data/processed/raw_qa.jsonl"))
    tokenizer_corpus_path = Path(paths["tokenizer_corpus"])

    indic_globs = ingestion.get("indic_jsonl_globs", [])
    indiccorp_txt_files = ingestion.get("indiccorp_txt_files", [])
    samanantar_pairs = ingestion.get("samanantar_pairs", [])
    gov_jsonl_globs = ingestion.get("gov_jsonl_globs", [])
    gov_csv_globs = ingestion.get("gov_csv_globs", [])
    qa_jsonl_globs = ingestion.get("gov_qa_jsonl_globs", [])
    gov_urls_file = Path(ingestion.get("gov_urls_file", "data/government/gov_urls.txt"))
    max_lines_per_file = ingestion.get("max_lines_per_file")
    max_records_per_language = ingestion.get("max_records_per_language")

    supported_languages = [normalize_language_code(language) for language in config.get("languages", {}).get("supported", ["en", "hi", "gu"])]

    raw_records: List[RawRecord] = []
    qa_records: List[QARecord] = []

    raw_records.extend(load_indic_jsonl(expand_globs(indic_globs), text_fields=("text", "sentence", "content")))

    for item in indiccorp_txt_files:
        path = Path(str(item.get("path", "")).strip())
        language = str(item.get("language", "auto"))
        if not str(path):
            continue
        raw_records.extend(load_indic_plain_text(path, language=language, max_lines=max_lines_per_file))

    for pair in samanantar_pairs:
        src_lang = str(pair.get("src", "en"))
        tgt_lang = str(pair.get("tgt", "hi"))
        globs = pair.get("globs", [])
        raw_records.extend(load_samanantar_tsv(expand_globs(globs), src_lang=src_lang, tgt_lang=tgt_lang))

    raw_records.extend(load_government_jsonl(expand_globs(gov_jsonl_globs)))
    raw_records.extend(load_government_csv(expand_globs(gov_csv_globs)))
    qa_records.extend(load_government_qa_jsonl(expand_globs(qa_jsonl_globs)))

    if args.scrape:
        urls = _load_urls(gov_urls_file)
        if urls:
            raw_records.extend(scrape_government_urls(urls, timeout=int(ingestion.get("scrape_timeout", 15))))

    raw_records = filter_records_by_languages(raw_records, supported_languages)
    qa_records = filter_qa_by_languages(qa_records, supported_languages)
    raw_records = deduplicate_records(raw_records)
    raw_records = cap_records_per_language(raw_records, max_records_per_language=max_records_per_language)

    output_raw_text.parent.mkdir(parents=True, exist_ok=True)
    output_raw_qa.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_raw_text, _records_to_jsonl(raw_records))
    write_jsonl(output_raw_qa, _qa_to_jsonl(qa_records))

    pretraining_pairs = build_pretraining_pairs(_records_to_jsonl(raw_records))
    finetuning_pairs = build_finetuning_pairs(_qa_to_jsonl(qa_records))

    save_pairs_jsonl(paths["pretrain_pairs"], pretraining_pairs)
    save_pairs_jsonl(paths["finetune_pairs"], finetuning_pairs)

    tokenizer_lines = [record.text for record in raw_records]
    if qa_records:
        tokenizer_lines.extend([record.question for record in qa_records])
        tokenizer_lines.extend([record.answer for record in qa_records])
    write_text(tokenizer_corpus_path, "\n".join(tokenizer_lines))

    print(
        f"Ingested {len(raw_records)} raw records, {len(qa_records)} QA records, "
        f"{len(pretraining_pairs)} pretrain pairs, {len(finetuning_pairs)} finetune pairs."
    )


if __name__ == "__main__":
    main()
