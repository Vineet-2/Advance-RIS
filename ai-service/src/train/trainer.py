import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import compute_multilabel_metrics, find_optimal_thresholds, save_metrics

logger = logging.getLogger(__name__)


class MultiLabelTrainer:
    """
    High-performance multi-label trainer with AMP, channels_last,
    warmup + cosine LR schedule, gradient clipping, macro-AUROC early stopping,
    and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str | Path,
        classes: List[str],
        test_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        warmup_epochs: int = 2,
        min_lr: float = 1e-6,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 5,
        amp: bool = True,
        channels_last: bool = True,
        dry_run: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes

        self.epochs = 1 if dry_run else epochs
        self.warmup_epochs = 0 if dry_run else warmup_epochs
        self.min_lr = min_lr
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.amp = amp and (device.type == "cuda")
        self.channels_last = channels_last and (device.type == "cuda")
        self.dry_run = dry_run

        self.base_lr = optimizer.param_groups[0]["lr"]
        if self.amp and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        elif self.amp and hasattr(torch.cuda.amp, "GradScaler"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        else:
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, opt): opt.step()
                def update(self): pass
                def unscale_(self, opt): pass
                def state_dict(self): return {}
                def load_state_dict(self, state): pass
            self.scaler = DummyScaler()

        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Scheduler setup
        cosine_epochs = max(1, self.epochs - self.warmup_epochs)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=self.min_lr
        )

        self.start_epoch = 1
        self.best_macro_auroc = -1.0
        self.patience_counter = 0

    def _autocast(self):
        if self.amp and self.device.type == "cuda":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", enabled=True)
            return torch.cuda.amp.autocast(enabled=True)
        from contextlib import nullcontext
        return nullcontext()

    def _adjust_warmup_lr(self, epoch: int):
        """Linearly ramps learning rate during warmup epochs."""
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / max(1, self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def _step_scheduler(self, epoch: int):
        if epoch > self.warmup_epochs:
            self.cosine_scheduler.step()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._adjust_warmup_lr(epoch)
        total_loss, total_samples = 0.0, 0

        for batch_idx, (images, targets) in enumerate(self.train_loader, start=1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.channels_last and images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                # Handle models returning (logits, fmaps) or logits
                output = self.model(images)
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx % 100 == 0 or batch_idx == len(self.train_loader):
                logger.info(
                    "Epoch %d [%d/%d] - Batch Loss: %.4f | Avg Loss: %.4f",
                    epoch, batch_idx, len(self.train_loader), loss.item(), total_loss / max(total_samples, 1)
                )

            if self.dry_run and batch_idx >= 2:
                logger.info("Dry-run: exiting train batch loop early (2 batches executed).")
                break

        self._step_scheduler(epoch)
        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_probs, all_targets = [], []

        for batch_idx, (images, targets) in enumerate(data_loader, start=1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.channels_last and images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

            with self._autocast():
                output = self.model(images)
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits, targets)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx % 50 == 0 or batch_idx == len(data_loader):
                logger.info("Evaluating [%d/%d] batches - Current Loss: %.4f", batch_idx, len(data_loader), loss.item())

            if self.dry_run and batch_idx >= 2:
                logger.info("Dry-run: exiting eval batch loop early (2 batches executed).")
                break

        avg_loss = total_loss / max(total_samples, 1)
        y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(self.classes)))
        y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, len(self.classes)))
        return avg_loss, y_true, y_probs

    def save_checkpoint(
        self,
        checkpoint_name: str,
        epoch: int,
        val_metrics: Dict[str, Any],
        val_loss: float,
    ) -> Path:
        save_path = self.save_dir / checkpoint_name
        ckpt = {
            "epoch": epoch,
            "architecture": self.model.__class__.__name__,
            "classes": self.classes,
            "num_classes": len(self.classes),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": val_metrics,
            "val_loss": val_loss,
            "best_macro_auroc": self.best_macro_auroc,
        }
        torch.save(ckpt, save_path)
        logger.info("Saved checkpoint to: %s", save_path)
        return save_path

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resumes optimizer, scheduler, epoch, and weights from an existing training checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and self.amp:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_macro_auroc = ckpt.get("best_macro_auroc", -1.0)
        logger.info("Resumed from %s at epoch %d (best_macro_auroc=%.4f)", checkpoint_path, self.start_epoch, self.best_macro_auroc)

    def fit(self) -> Dict[str, Any]:
        logger.info("Starting training run on device %s for %d epochs...", self.device, self.epochs)
        best_val_metrics: Dict[str, Any] = {}

        for epoch in range(self.start_epoch, self.epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_targets, val_probs = self.evaluate(self.val_loader)

            # Find optimal thresholds and compute comprehensive validation metrics
            optimal_thresholds = find_optimal_thresholds(val_targets, val_probs, self.classes)
            val_metrics = compute_multilabel_metrics(val_targets, val_probs, self.classes, optimal_thresholds)
            macro_auroc = val_metrics["macro_auroc"]
            macro_f1 = val_metrics["macro_f1"]

            logger.info(
                "Epoch %d/%d [lr=%.6f] - Train Loss: %.4f | Val Loss: %.4f | Val Macro-AUROC: %.4f | Val Macro-F1: %.4f",
                epoch, self.epochs, current_lr, train_loss, val_loss, macro_auroc, macro_f1
            )

            # Check for best model
            if macro_auroc > self.best_macro_auroc:
                self.best_macro_auroc = macro_auroc
                self.patience_counter = 0
                best_val_metrics = copy.deepcopy(val_metrics)
                self.save_checkpoint("best_model.pth", epoch, val_metrics, val_loss)
                save_metrics(val_metrics, self.save_dir, prefix="best_val_metrics")
            else:
                self.patience_counter += 1

            # Always save last checkpoint for resume
            self.save_checkpoint("last_model.pth", epoch, val_metrics, val_loss)

            if not self.dry_run and self.patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", self.early_stopping_patience)
                break

        # Optional test evaluation
        if self.test_loader and best_val_metrics:
            logger.info("Evaluating best model on test set...")
            best_ckpt = self.save_dir / "best_model.pth"
            if best_ckpt.is_file():
                ckpt = torch.load(best_ckpt, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
            test_loss, test_targets, test_probs = self.evaluate(self.test_loader)
            test_metrics = compute_multilabel_metrics(
                test_targets, test_probs, self.classes, best_val_metrics.get("thresholds")
            )
            logger.info("Test Macro-AUROC: %.4f | Test Macro-F1: %.4f", test_metrics["macro_auroc"], test_metrics["macro_f1"])
            save_metrics(test_metrics, self.save_dir, prefix="test_metrics")

        return best_val_metrics
