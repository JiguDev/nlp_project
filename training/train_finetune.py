"""Fine-tune the multilingual transformer on government Q&A data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.data_pipeline import build_finetuning_pairs, load_government_qa, save_pairs_jsonl
from training.dataset import SequencePairDataset
from training.engine import build_dataloader, build_model, get_device, load_checkpoint, save_checkpoint, train_one_epoch
from utils.config import load_config
from utils.metrics import perplexity_from_loss
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune the multilingual transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML configuration file")
    return parser.parse_args()


def ensure_finetune_data(config: dict) -> Path:
    """Prepare the fine-tuning dataset if it does not exist."""
    data_dir = Path(config["paths"]["data_dir"])
    finetune_path = Path(config["paths"]["finetune_pairs"])
    if finetune_path.exists():
        return finetune_path
    qa_path = data_dir / "dummy" / "qa_pairs.jsonl"
    qa_records = load_government_qa(qa_path) if qa_path.exists() else []
    pairs = build_finetuning_pairs(qa_records)
    return save_pairs_jsonl(finetune_path, pairs)


def main() -> None:
    """Run the fine-tuning loop."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    tokenizer_model_path = Path(config["paths"]["tokenizer_model"])
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_model_path}")

    finetune_path = ensure_finetune_data(config)
    tokenizer = SentencePieceTokenizer(tokenizer_model_path)
    dataset = SequencePairDataset.from_jsonl(
        finetune_path,
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
    checkpoint_path = Path(config["paths"]["pretrain_checkpoint"])
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, optimizer, map_location=device)

    for epoch in range(int(config["training"]["finetune_epochs"])):
        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device,
            tokenizer.pad_id,
            label_smoothing=float(config["training"]["label_smoothing"]),
            grad_clip=float(config["training"]["grad_clip"]),
        )
        print(f"Fine-tune epoch {epoch + 1}: loss={loss:.4f}, perplexity={perplexity_from_loss(loss):.2f}")
        save_checkpoint(config["paths"]["finetune_checkpoint"], model, optimizer, epoch + 1)


if __name__ == "__main__":
    main()
