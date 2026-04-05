"""Reusable training utilities for the multilingual chatbot."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.transformer import TransformerConfig, TransformerSeq2Seq
from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.dataset import SequencePairDataset, collate_sequence_pairs
from utils.config import load_config


def build_tokenizer(model_path: str | Path) -> SentencePieceTokenizer:
    """Load the shared SentencePiece tokenizer."""
    return SentencePieceTokenizer(model_path)


def build_model(tokenizer: SentencePieceTokenizer, model_settings: dict) -> TransformerSeq2Seq:
    """Construct a transformer with tokenizer-aware vocabulary settings."""
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=int(model_settings["d_model"]),
        num_heads=int(model_settings["num_heads"]),
        num_encoder_layers=int(model_settings["num_encoder_layers"]),
        num_decoder_layers=int(model_settings["num_decoder_layers"]),
        dim_feedforward=int(model_settings["dim_feedforward"]),
        dropout=float(model_settings["dropout"]),
        max_len=int(model_settings["max_len"]),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        tie_embeddings=bool(model_settings.get("tie_embeddings", True)),
    )
    return TransformerSeq2Seq(config)


def build_dataloader(dataset: SequencePairDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a dataloader with a padding-aware collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_sequence_pairs(batch, dataset.tokenizer.pad_id),
    )


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "cpu":
        print("Using CPU")
        return torch.device("cpu")

    if torch.cuda.is_available():
        print("✅ CUDA available → Using GPU")
        return torch.device("cuda")

    print("⚠️ CUDA not available → Using CPU")
    return torch.device("cpu")


def sequence_loss(logits: torch.Tensor, target_tokens: torch.Tensor, pad_id: int, label_smoothing: float = 0.0) -> torch.Tensor:
    """Compute token-level cross-entropy loss for seq2seq generation."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    return criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))


def train_one_epoch(
    model: TransformerSeq2Seq,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    pad_id: int,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
) -> float:
    """Run a single training or evaluation epoch."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        src_tokens = batch["src_tokens"].to(device)
        tgt_tokens = batch["tgt_tokens"].to(device)
        decoder_input = tgt_tokens[:, :-1]
        decoder_target = tgt_tokens[:, 1:]
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(src_tokens, decoder_input)
        loss = sequence_loss(logits, decoder_target, pad_id, label_smoothing=label_smoothing)
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_checkpoint(path: str | Path, model: TransformerSeq2Seq, optimizer: Optional[torch.optim.Optimizer], epoch: int, extra_state: Optional[dict] = None) -> Path:
    """Save a model checkpoint with optimizer and metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "epoch": epoch, "model_config": model.export_config()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, model: TransformerSeq2Seq, optimizer: Optional[torch.optim.Optimizer] = None, map_location: str | torch.device = "cpu") -> dict:
    """Load model weights and optionally optimizer state."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
