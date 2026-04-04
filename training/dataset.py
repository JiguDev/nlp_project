"""Dataset and collation utilities for seq2seq transformer training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from tokenizer.sp_tokenizer import SentencePieceTokenizer
from utils.io import iter_jsonl


@dataclass
class SequencePair:
    """A single source-target example used for training."""

    source: str
    target: str
    language: str = "en"


class SequencePairDataset(Dataset):
    """Torch dataset backed by JSONL sequence pairs."""

    def __init__(self, records: Sequence[dict], tokenizer: SentencePieceTokenizer, max_source_length: int, max_target_length: int) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: SentencePieceTokenizer, max_source_length: int, max_target_length: int) -> "SequencePairDataset":
        """Load a dataset from a JSONL file containing source and target fields."""
        return cls(list(iter_jsonl(path)), tokenizer, max_source_length, max_target_length)

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str, max_length: int) -> List[int]:
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        return token_ids[:max_length]

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        source = str(record.get("source", record.get("question", "")))
        target = str(record.get("target", record.get("answer", "")))
        language = str(record.get("language", "en"))
        source_ids = self._encode(source, self.max_source_length)
        target_ids = self._encode(target, self.max_target_length)
        return {"source_ids": source_ids, "target_ids": target_ids, "language": language}


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    """Pad variable-length token sequences into a dense tensor."""
    max_length = max(len(sequence) for sequence in sequences)
    output = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        output[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return output


def collate_sequence_pairs(batch: Sequence[dict], pad_id: int) -> dict:
    """Collate function producing padded source and target tensors."""
    source_ids = [item["source_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    src_tokens = pad_sequences(source_ids, pad_id)
    tgt_tokens = pad_sequences(target_ids, pad_id)
    return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}
