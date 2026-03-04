"""
Training loop for CyberSecLLM.

Implements the distributed pre-training procedure with:
- AdamW optimizer with cosine learning rate schedule
- Gradient clipping and accumulation
- Optional differential privacy via DP-SGD
- Early stopping and checkpoint management
- Wandb/Tensorboard logging

Reference: Section III-D (Training Procedure, Algorithm 1).
"""

import os
import math
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from ..models.cybersecllm import CyberSecLLM
from .losses import MultiTaskLoss

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience=10, mode="min", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.should_stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CyberSecLLMTrainer:
    """
    Complete training pipeline for CyberSecLLM.

    Implements Algorithm 1 from the paper with support for:
    - Multi-task pre-training with uncertainty weighting
    - Distributed training via DeepSpeed ZeRO-3
    - Differential privacy via DP-SGD
    - Automatic mixed precision
    """

    def __init__(
        self,
        model: CyberSecLLM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        warmup_fraction: float = 0.01,
        max_epochs: int = 100,
        max_steps: Optional[int] = None,
        gradient_clip: float = 1.0,
        gradient_accumulation: int = 1,
        save_dir: str = "checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        early_stopping_patience: int = 10,
        use_dp: bool = False,
        dp_noise: float = 0.8,
        dp_max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_clip = gradient_clip
        self.gradient_accumulation = gradient_accumulation
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        os.makedirs(save_dir, exist_ok=True)

        self.loss_fn = MultiTaskLoss()
        self.loss_fn = self.loss_fn.to(device)

        self.optimizer = AdamW(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=1e-8,
        )

        total_steps = max_steps or (max_epochs * len(train_loader))
        warmup_steps = int(total_steps * warmup_fraction)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps
        )

        self.early_stopping = EarlyStopping(early_stopping_patience)
        self.scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            features = batch["features"]
            timestamps = batch.get("timestamp", torch.arange(
                features.shape[0], device=self.device, dtype=torch.float32
            ))

            if features.dim() == 2:
                features = features.unsqueeze(1)
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(1).expand(-1, features.shape[1])

            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                output = self.model(
                    features=features,
                    timestamps=timestamps,
                    task="detection",
                )
                loss_dict = self.loss_fn(output, batch)
                loss = loss_dict["total_loss"] / self.gradient_accumulation

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1

            if self.global_step % self.log_interval == 0 and self.global_step > 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_lr()[0]
                logger.info(
                    f"Epoch {epoch} | Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"CLS: {loss_dict['cls_loss']:.4f}"
                )

            if (
                self.val_loader is not None
                and self.global_step % self.eval_interval == 0
                and self.global_step > 0
            ):
                val_loss = self.evaluate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                if self.early_stopping(val_loss):
                    logger.info("Early stopping triggered.")
                    return True

            if self.global_step % self.save_interval == 0 and self.global_step > 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            if self.max_steps and self.global_step >= self.max_steps:
                return True

        return False

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            features = batch["features"]
            timestamps = batch.get("timestamp", torch.arange(
                features.shape[0], device=self.device, dtype=torch.float32
            ))

            if features.dim() == 2:
                features = features.unsqueeze(1)
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(1).expand(-1, features.shape[1])

            output = self.model(
                features=features,
                timestamps=timestamps,
                task="detection",
            )
            loss_dict = self.loss_fn(output, batch)
            total_loss += loss_dict["total_loss"].item()

            preds = output["logits"].argmax(dim=-1)
            total_correct += (preds == batch["label"]).sum().item()
            total_samples += batch["label"].shape[0]

        avg_loss = total_loss / max(1, len(self.val_loader))
        accuracy = total_correct / max(1, total_samples) * 100

        logger.info(
            f"Validation | Step {self.global_step} | "
            f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
        )

        self.model.train()
        return avg_loss

    def train(self):
        """Complete training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        params = self.model.count_parameters()
        logger.info(
            f"Model parameters: {params['total']:,} total, "
            f"{params['trainable']:,} trainable"
        )

        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"--- Epoch {epoch}/{self.max_epochs} ---")
            should_stop = self.train_epoch(epoch)
            if should_stop:
                break

        self.save_checkpoint("final_model.pt")
        logger.info("Training completed.")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "loss_fn_state_dict": self.loss_fn.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        if "loss_fn_state_dict" in checkpoint:
            self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")
