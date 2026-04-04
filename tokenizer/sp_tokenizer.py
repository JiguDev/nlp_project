"""SentencePiece tokenizer wrapper with language-token support."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import sentencepiece as spm
except Exception as exc:  # pragma: no cover - dependency guard
    spm = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class TokenizerConfig:
    """Configuration used to train a shared multilingual tokenizer."""

    model_prefix: str
    vocab_size: int = 32000
    model_type: str = "unigram"
    character_coverage: float = 0.9995
    user_defined_symbols: Sequence[str] = ("<en>", "<hi>", "<gu>")


class SentencePieceTokenizer:
    """Thin wrapper around SentencePiece for multilingual sequence modeling."""

    def __init__(self, model_file: str | Path) -> None:
        if spm is None:
            raise ImportError("sentencepiece is required to use SentencePieceTokenizer") from _IMPORT_ERROR
        self.model_file = Path(model_file)
        self.processor = spm.SentencePieceProcessor()
        self.processor.Load(str(self.model_file))

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text into token IDs."""
        return list(self.processor.EncodeAsIds(text, add_bos=add_bos, add_eos=add_eos))

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs back to text."""
        return self.processor.DecodeIds(list(token_ids))

    def tokenize(self, text: str) -> List[str]:
        """Return SentencePiece tokens for inspection or debugging."""
        return list(self.processor.EncodeAsPieces(text))

    def save_pretrained(self, destination_dir: str | Path) -> Path:
        """Copy the underlying model files into a directory if needed."""
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model_file = destination / "tokenizer.model"
        model_file.write_bytes(self.model_file.read_bytes())
        return model_file


def train_sentencepiece(input_file: str | Path, config: TokenizerConfig) -> Path:
    """Train a shared multilingual tokenizer from a plain-text corpus."""
    if spm is None:
        raise ImportError("sentencepiece is required to train the tokenizer") from _IMPORT_ERROR
    model_prefix = str(Path(config.model_prefix))
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=model_prefix,
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        user_defined_symbols=list(config.user_defined_symbols),
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        hard_vocab_limit=False,
    )
    return Path(f"{model_prefix}.model")
