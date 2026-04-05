"""Pretrain the multilingual transformer on autoencoding corpora."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from tokenizer.sp_tokenizer import SentencePieceTokenizer
from training.dataset import SequencePairDataset
from training.engine import build_dataloader, build_model, get_device, save_checkpoint, train_one_epoch
from utils.config import load_config
from utils.metrics import perplexity_from_loss
from utils.seed import set_seed


# =========================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pretrain the multilingual transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML configuration file")
    return parser.parse_args()


# =========================
def ensure_artifacts(config: dict) -> tuple[Path, Path]:
    """
    Ensure tokenizer and dataset exist.
    DO NOT rebuild anything unnecessarily.
    """

    processed_dir = Path(config["paths"]["processed_dir"])
    tokenizer_model_prefix = Path(config["paths"]["tokenizer_model_prefix"])
    pretrain_pairs_path = Path(config["paths"]["pretrain_pairs"])

    tokenizer_model_path = Path(f"{tokenizer_model_prefix}.model")

    # 🔥 Check tokenizer
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            "❌ Tokenizer not found. Please run: python -m tokenizer.train_tokenizer"
        )

    # 🔥 Check dataset
    if not pretrain_pairs_path.exists():
        raise FileNotFoundError(
            "❌ pretrain_pairs.jsonl not found. Please run: python -m training.data_pipeline"
        )

    return tokenizer_model_path, pretrain_pairs_path


# =========================
def main() -> None:
    """Run the pretraining loop."""

    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config.get("seed", 42)))

    # =========================
    # Load artifacts
    tokenizer_model_path, dataset_path = ensure_artifacts(config)

    print("\n==============================")
    print("Loading tokenizer and dataset...")
    print("==============================\n")

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

    # =========================
    # Device setup
    device = get_device(str(config["training"]["device"]))
    print(f"Using device: {device}\n")

    # =========================
    # Build model
    model = build_model(tokenizer, config["model"]).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    # =========================
    # Training loop
    print("==============================")
    print("🚀 Starting Pretraining...")
    print("==============================\n")

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

        ppl = perplexity_from_loss(loss)

        print(f"Epoch {epoch + 1}: loss={loss:.4f}, perplexity={ppl:.2f}")

        save_checkpoint(
            config["paths"]["pretrain_checkpoint"],
            model,
            optimizer,
            epoch + 1
        )

    print("\n==============================")
    print("✅ TRAINING COMPLETE")
    print("==============================\n")


# =========================
if __name__ == "__main__":
    main()