import argparse
import copy
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

# Add ai-service to sys.path
AI_SERVICE_DIR = Path(__file__).resolve().parent.parent.parent
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from src.data.dataset import NIHChestXrayDataset, NIH_CLASSES
from src.data.splits import parse_nih_csv, create_patient_splits, assert_no_patient_leakage
from src.train.losses import get_loss_fn, compute_pos_weights
from src.train.trainer import MultiLabelTrainer
from src.models.densenet import build_densenet121

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("src.train")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "base_config" in cfg and cfg["base_config"]:
        candidate_rel = cfg_file.parent / cfg["base_config"]
        candidate_cwd = Path(cfg["base_config"])
        if candidate_rel.is_file():
            base_cfg = load_config(candidate_rel)
        elif candidate_cwd.is_file():
            base_cfg = load_config(candidate_cwd)
        else:
            raise FileNotFoundError(f"Base config '{cfg['base_config']}' not found from {cfg_file.parent} or cwd")
        cfg = deep_merge(base_cfg, cfg)

    return cfg


def build_model(model_cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    arch = model_cfg.get("architecture", "densenet121").lower()
    pretrained = model_cfg.get("pretrained", True)

    if "densenet" in arch:
        return build_densenet121(num_classes=num_classes, pretrained=pretrained)
    elif "resnet" in arch:
        from src.models.resnet import build_resnet50
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif "swin" in arch:
        from src.models.swin import build_swin_tiny
        return build_swin_tiny(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")


def main():
    parser = argparse.ArgumentParser(description="Retrain NIH ChestX-ray14 Multi-Label Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch with 2 batches to test execution")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    classes = cfg.get("classes", NIH_CLASSES)
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Retraining entry point running on device: %s (dry_run=%s)", device, args.dry_run)

    # 1. Parse data records
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    csv_path = data_cfg.get("csv_path", "test_nih/Data_Entry_2017_v2020.csv")
    images_dir = data_cfg.get("images_dir", "test_nih/images")

    # Resolve paths relative to ai-service if needed
    if not os.path.isabs(csv_path):
        csv_path = str(AI_SERVICE_DIR / csv_path)
    if not os.path.isabs(images_dir):
        images_dir = str(AI_SERVICE_DIR / images_dir)

    train_records, val_records, test_records = [], [], []
    if os.path.isfile(csv_path) and os.path.isdir(images_dir):
        all_records = parse_nih_csv(csv_path, images_dir)
        train_records, val_records, test_records = create_patient_splits(
            all_records,
            val_fraction=data_cfg.get("val_fraction", 0.1),
            test_fraction=data_cfg.get("test_fraction", 0.1),
            seed=seed,
            train_val_list_path=data_cfg.get("train_val_list_path"),
            test_list_path=data_cfg.get("test_list_path"),
        )
    elif args.dry_run:
        logger.warning("Images dir (%s) not found. Generating synthetic mock records for dry-run verification.", images_dir)
        synthetic_records = [
            {"image_name": f"mock_{i}.png", "path": "mock", "patient_id": f"P{i//2}", "labels": [classes[i % num_classes]]}
            for i in range(20)
        ]
        train_records, val_records, test_records = create_patient_splits(
            synthetic_records, val_fraction=0.3, test_fraction=0.2, seed=seed
        )
    else:
        raise FileNotFoundError(f"Missing dataset files: CSV={csv_path}, ImagesDir={images_dir}. Use --dry-run for testing.")

    assert_no_patient_leakage(train_records, val_records, test_records if test_records else None)

    # 2. Build Datasets & DataLoaders
    image_size = data_cfg.get("image_size", 224)
    batch_size = train_cfg.get("batch_size", 32)
    workers = data_cfg.get("num_workers", 0)
    pin_mem = data_cfg.get("pin_memory", True) and (device.type == "cuda")

    train_ds = NIHChestXrayDataset(train_records, classes=classes, is_training=True, image_size=image_size)
    val_ds = NIHChestXrayDataset(val_records, classes=classes, is_training=False, image_size=image_size)
    test_ds = NIHChestXrayDataset(test_records, classes=classes, is_training=False, image_size=image_size) if test_records else None

    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": pin_mem,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs) if test_ds else None

    # 3. Model construction
    model_cfg = cfg.get("model", {"architecture": "densenet121", "pretrained": True})
    model = build_model(model_cfg, num_classes=num_classes).to(device)

    # Optional torch.compile
    if train_cfg.get("torch_compile", False) and hasattr(torch, "compile"):
        try:
            logger.info("Applying torch.compile(model)...")
            model = torch.compile(model)
        except Exception as exc:
            logger.warning("torch.compile failed: %s; falling back to uncompiled model", exc)

    # 4. Loss & Optimizer
    loss_type = train_cfg.get("loss_type", "bce_pos_weight")
    pos_weight = None
    if loss_type == "bce_pos_weight":
        pos_weight = compute_pos_weights(
            train_records,
            num_classes=num_classes,
            classes=classes,
            max_weight=train_cfg.get("max_pos_weight", 50.0),
        )

    criterion = get_loss_fn(
        loss_type=loss_type,
        pos_weight=pos_weight,
        gamma=train_cfg.get("focal_gamma", 2.0),
        device=device,
    )

    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. Trainer
    save_dir = output_cfg.get("save_dir", "outputs/retrained")
    if not os.path.isabs(save_dir):
        save_dir = str(AI_SERVICE_DIR / save_dir)

    trainer = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        classes=classes,
        epochs=train_cfg.get("epochs", 20),
        warmup_epochs=train_cfg.get("warmup_epochs", 2),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
        amp=train_cfg.get("amp", True),
        channels_last=train_cfg.get("channels_last", True),
        dry_run=args.dry_run,
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    results = trainer.fit()
    logger.info("Training script execution finished. Results summary: %s", results)


if __name__ == "__main__":
    main()
