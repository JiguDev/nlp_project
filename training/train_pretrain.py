"""Pretrain the multilingual transformer on autoencoding corpora."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.data_pipeline import build_pretraining_pairs, build_tokenizer_corpus, load_plain_text_corpus, save_pairs_jsonl
from training.dataset import SequencePairDataset
from training.engine import build_dataloader, build_model, get_device, save_checkpoint, train_one_epoch
from utils.config import load_config
from utils.metrics import perplexity_from_loss
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pretrain the multilingual transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML configuration file")
    return parser.parse_args()


def ensure_artifacts(config: dict) -> tuple[Path, Path]:
    """Build the tokenizer corpus and pretraining pairs if needed."""
    data_dir = Path(config["paths"]["data_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    tokenizer_corpus = Path(config["paths"]["tokenizer_corpus"])
    tokenizer_model_prefix = Path(config["paths"]["tokenizer_model_prefix"])
    pretrain_pairs_path = Path(config["paths"]["pretrain_pairs"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not tokenizer_corpus.exists():
        build_tokenizer_corpus(data_dir, tokenizer_corpus)
    if not pretrain_pairs_path.exists():
        multilingual_path = data_dir / "dummy" / "multilingual_corpus.txt"
        records = load_plain_text_corpus(multilingual_path, language="mixed") if multilingual_path.exists() else []
        pairs = build_pretraining_pairs(records)
        save_pairs_jsonl(pretrain_pairs_path, pairs)
    if not Path(f"{tokenizer_model_prefix}.model").exists():
        from tokenizer.sp_tokenizer import TokenizerConfig, train_sentencepiece

        vocab_size = int(config.get("tokenizer", {}).get("vocab_size", 2000))
        train_sentencepiece(tokenizer_corpus, TokenizerConfig(model_prefix=str(tokenizer_model_prefix), vocab_size=vocab_size))
    return Path(f"{tokenizer_model_prefix}.model"), pretrain_pairs_path


def main() -> None:
    """Run the pretraining loop."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    tokenizer_model_path, dataset_path = ensure_artifacts(config)
    tokenizer = SentencePieceTokenizer(tokenizer_model_path)
    dataset = SequencePairDataset.from_jsonl(
        dataset_path,
        tokenizer=tokenizer,
        max_source_length=int(config["training"]["max_source_length"]),
        max_target_length=int(config["training"]["max_target_length"]),
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )
    device = get_device(str(config["training"]["device"]))
    model = build_model(tokenizer, config["model"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))

    for epoch in range(int(config["training"]["pretrain_epochs"])):
        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
            tokenizer.pad_id,
            label_smoothing=float(config["training"]["label_smoothing"]),
            grad_clip=float(config["training"]["grad_clip"]),
        )
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, perplexity={perplexity_from_loss(loss):.2f}")
        save_checkpoint(config["paths"]["pretrain_checkpoint"], model, optimizer, epoch + 1)


if __name__ == "__main__":
    main()
