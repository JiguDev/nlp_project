"""Train the shared SentencePiece tokenizer on multilingual and government text."""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.sp_tokenizer import TokenizerConfig, train_sentencepiece
from training.data_pipeline import build_tokenizer_corpus


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tokenizer training."""
    parser = argparse.ArgumentParser(description="Train multilingual SentencePiece tokenizer")
    parser.add_argument("--input-dir", type=str, default="data", help="Root directory containing raw corpora")
    parser.add_argument("--output-corpus", type=str, default="data/processed/tokenizer_corpus.txt", help="Merged corpus for tokenizer training")
    parser.add_argument("--model-prefix", type=str, default="data/processed/indic_gov_tokenizer", help="Output prefix for the trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Tokenizer vocabulary size")
    return parser.parse_args()


def main() -> None:
    """Build the tokenizer corpus and train SentencePiece."""
    args = parse_args()
    corpus_path = build_tokenizer_corpus(Path(args.input_dir), Path(args.output_corpus))
    config = TokenizerConfig(model_prefix=args.model_prefix, vocab_size=args.vocab_size)
    model_path = train_sentencepiece(corpus_path, config)
    print(f"Tokenizer trained at: {model_path}")


if __name__ == "__main__":
    main()
