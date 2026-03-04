"""
Training script for CyberSecLLM.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --variant 7B --epochs 100 --lr 1e-3

For distributed training:
    torchrun --nproc_per_node=8 scripts/train.py --config configs/training_config.yaml
"""

import os
import sys
import argparse
import logging
import yaml

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.cybersecllm import CyberSecLLM, CyberSecLLMConfig
from src.data.dataset import SecurityDataset, IntegratedSecurityCorpus
from src.training.trainer import CyberSecLLMTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_data(config):
    """Set up training and validation data loaders."""
    data_root = config.get("data", {}).get("root", "data/")

    train_dataset = IntegratedSecurityCorpus(
        data_root=data_root,
        split="train",
        max_seq_len=config.get("data", {}).get("max_seq_len", 512),
    )

    val_dataset = IntegratedSecurityCorpus(
        data_root=data_root,
        split="val",
        max_seq_len=config.get("data", {}).get("max_seq_len", 512),
    )

    batch_size = config.get("training", {}).get("batch_size", 256)

    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size, num_workers=4, shuffle=True
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size, num_workers=4, shuffle=False
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train CyberSecLLM")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--variant", type=str, default="7B", choices=["3B", "7B", "13B"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {"training": {}, "data": {}}

    training_config = config.get("training", {})
    epochs = args.epochs or training_config.get("epochs", 100)
    lr = args.lr or training_config.get("optimizer", {}).get("lr", 1e-3)
    batch_size = args.batch_size or training_config.get("batch_size", 256)

    logger.info(f"Creating CyberSecLLM-{args.variant} model...")
    model = CyberSecLLM.from_variant(args.variant)
    params = model.count_parameters()
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")

    logger.info("Setting up data loaders...")
    train_loader, val_loader = setup_data(config)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    trainer = CyberSecLLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        max_epochs=epochs,
        gradient_clip=training_config.get("gradient_clip", 1.0),
        gradient_accumulation=training_config.get("gradient_accumulation_steps", 1),
        save_dir=args.save_dir,
        device=args.device,
    )

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
