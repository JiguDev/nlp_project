"""Fine-tune the multilingual transformer on government Q&A data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.dataset import SequencePairDataset
from training.engine import (
    build_dataloader,
    build_model,
    get_device,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from utils.config import load_config
from utils.metrics import perplexity_from_loss
from utils.seed import set_seed


# =========================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune the multilingual transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML configuration file")
    return parser.parse_args()


# =========================
def ensure_finetune_data(config: dict) -> Path:
    """
    Use already prepared fine-tuning dataset.
    (NO dummy fallback — must use real data)
    """
    finetune_path = Path(config["paths"]["finetune_pairs"])

    if not finetune_path.exists():
        raise FileNotFoundError(
            "❌ finetune_pairs.jsonl not found. Please run: python -m training.data_pipeline"
        )

    print(f"✅ Using fine-tuning data: {finetune_path}")
    return finetune_path


# =========================
def main() -> None:
    """Run the fine-tuning loop."""

    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config.get("seed", 42)))

    print("\n==============================")
    print("🔥 Starting Fine-tuning")
    print("==============================\n")

    # =========================
    # Tokenizer
    tokenizer_model_path = Path(config["paths"]["tokenizer_model"])

    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"❌ Tokenizer model not found: {tokenizer_model_path}"
        )

    tokenizer = SentencePieceTokenizer(tokenizer_model_path)

    # =========================
    # Dataset
    finetune_path = ensure_finetune_data(config)

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

    # =========================
    # Device
    device = get_device(str(config["training"]["device"]))
    print(f"Using device: {device}\n")

    # =========================
    # Model
    model = build_model(tokenizer, config["model"]).to(device)

    # =========================
    # Load pretrained weights
    checkpoint_path = Path(config["paths"]["pretrain_checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "❌ Pretrained checkpoint not found. Run pretraining first."
        )

    print(f"📦 Loading pretrained model: {checkpoint_path}")

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # 🔥 LOWER LR for finetuning
        weight_decay=float(config["training"]["weight_decay"]),
    )

    load_checkpoint(checkpoint_path, model, optimizer, map_location=device)

    # =========================
    # Training loop
    print("\n==============================")
    print("🚀 Fine-tuning in progress...")
    print("==============================\n")

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

        ppl = perplexity_from_loss(loss)

        print(f"Fine-tune Epoch {epoch + 1}: loss={loss:.4f}, perplexity={ppl:.2f}")

        save_checkpoint(
            config["paths"]["finetune_checkpoint"],
            model,
            optimizer,
            epoch + 1
        )

    print("\n==============================")
    print("✅ FINE-TUNING COMPLETE")
    print("==============================\n")


# =========================
if __name__ == "__main__":
    main()