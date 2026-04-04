"""Train the shared SentencePiece tokenizer on multilingual and government text."""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.sp_tokenizer import TokenizerConfig, train_sentencepiece
from training.data_pipeline import build_tokenizer_corpus
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tokenizer training."""
    parser = argparse.ArgumentParser(description="Train multilingual SentencePiece tokenizer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--input-dir", type=str, default="data", help="Root directory containing raw corpora")
    parser.add_argument("--output-corpus", type=str, default="data/processed/tokenizer_corpus.txt", help="Merged corpus for tokenizer training")
    parser.add_argument("--model-prefix", type=str, default="data/processed/indic_gov_tokenizer", help="Output prefix for the trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=None, help="Tokenizer vocabulary size (overrides config)")
    return parser.parse_args()


def main() -> None:
    """Build the tokenizer corpus and train SentencePiece."""
    args = parse_args()
    config = load_config(args.config)
    supported_languages = config.get("languages", {}).get("supported", ["en", "hi", "gu"])
    output_corpus = Path(config.get("paths", {}).get("tokenizer_corpus", args.output_corpus))
    model_prefix = str(config.get("paths", {}).get("tokenizer_model_prefix", args.model_prefix))
    vocab_size = int(args.vocab_size or config.get("tokenizer", {}).get("vocab_size", 2000))
    model_type = str(config.get("tokenizer", {}).get("model_type", "unigram"))
    corpus_path = build_tokenizer_corpus(Path(args.input_dir), output_corpus, supported_languages=supported_languages)
    sp_config = TokenizerConfig(model_prefix=model_prefix, vocab_size=vocab_size, model_type=model_type)
    model_path = train_sentencepiece(corpus_path, sp_config)
    print(f"Tokenizer trained at: {model_path}")


if __name__ == "__main__":
    main()
