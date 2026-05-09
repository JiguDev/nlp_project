"""Train the shared SentencePiece tokenizer on multilingual and government text."""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.sp_tokenizer import TokenizerConfig, train_sentencepiece
from utils.config import load_config
from utils.text import clean_text
from utils.lang import add_language_token


# =========================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tokenizer training."""
    parser = argparse.ArgumentParser(description="Train multilingual SentencePiece tokenizer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--input-dir", type=str, default="data", help="Root directory containing raw corpora")
    parser.add_argument("--output-corpus", type=str, default="data/processed/tokenizer_corpus.txt", help="Merged corpus for tokenizer training")
    parser.add_argument("--model-prefix", type=str, default="data/processed/indic_gov_tokenizer", help="Output prefix for the trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=None, help="Tokenizer vocabulary size (overrides config)")
    return parser.parse_args()


# =========================
def build_tokenizer_corpus(data_dir: Path, output_path: Path) -> Path:
    """
    Build a plain text corpus for tokenizer training.
    Each line = one sentence with language token.
    """

    print("\n==============================")
    print("Building tokenizer corpus...")
    print("==============================\n")

    texts = []

    # Dataset paths
    en_path = data_dir / "multilingual" / "indiccorp" / "en.txt"
    hi_path = data_dir / "multilingual" / "indiccorp" / "hi-1.txt"
    gu_path = data_dir / "multilingual" / "indiccorp" / "gu.txt"

    LIMIT = 10000  # Safe for tokenizer

    def process(path: Path, lang: str):
        if not path.exists():
            print(f"⚠️ Skipping missing file: {path}")
            return

        print(f"Processing {path.name} ({lang})...")

        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= LIMIT:
                    break

                text = clean_text(line)

                if text:
                    text = add_language_token(text, lang)
                    texts.append(text)

                if i % 1000 == 0:
                    print(f"{path.name}: {i} lines processed")

        print(f"Finished {path.name}\n")

    # Process all languages
    process(en_path, "en")
    process(hi_path, "hi")
    process(gu_path, "gu")

    if not texts:
        dummy_path = data_dir / "dummy" / "multilingual_corpus.txt"
        if dummy_path.exists():
            print(f"Fallback to dummy corpus: {dummy_path}")
            # Just read it all as mixed or en for dummy purposes
            process(dummy_path, "en")

    # Save corpus
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    print("==============================")
    print(f"Corpus saved: {output_path}")
    print(f"Total lines: {len(texts)}")
    print("==============================\n")

    return output_path


# =========================
def main() -> None:
    """Build tokenizer corpus and train SentencePiece."""
    args = parse_args()

    config = load_config(args.config)

    output_corpus = Path(config.get("paths", {}).get("tokenizer_corpus", args.output_corpus))
    model_prefix = str(config.get("paths", {}).get("tokenizer_model_prefix", args.model_prefix))

    vocab_size = int(args.vocab_size or config.get("tokenizer", {}).get("vocab_size", 8000))
    model_type = str(config.get("tokenizer", {}).get("model_type", "unigram"))

    # 🔥 Build corpus
    corpus_path = build_tokenizer_corpus(Path(args.input_dir), output_corpus)

    # 🔥 Train tokenizer
    print("Training SentencePiece tokenizer...\n")

    sp_config = TokenizerConfig(
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type
    )

    model_path = train_sentencepiece(corpus_path, sp_config)

    print("\n==============================")
    print("✅ TOKENIZER TRAINED")
    print("==============================")
    print(f"Model: {model_path}\n")


# =========================
if __name__ == "__main__":
    main()