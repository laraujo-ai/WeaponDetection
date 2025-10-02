import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from super_gradients.training import models, Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Train detector with SuperGradients.")
    p.add_argument("--data-dir", required=True, type=str, help="Root dataset folder.")
    p.add_argument(
        "--classes",
        required=True,
        type=str,
        help="Comma-separated class names (e.g. 'weapon,person').",
    )
    p.add_argument("--model-arch", type=str, default="yolo_nas_s", help="Model architecture name.")
    p.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    p.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output directory.")
    p.add_argument("--experiment-name", type=str, default="sg_experiment", help="Experiment name.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers.")
    return p.parse_args()


def prepare_dataset_params(data_dir: str, classes: List[str]) -> dict:
    """Build dataset parameter dictionary for SuperGradients dataloaders."""
    data_dir = str(Path(data_dir).expanduser())
    return {
        "data_dir": data_dir,
        "train_images_dir": "train/images",
        "train_labels_dir": "train/labels",
        "val_images_dir": "valid/images",
        "val_labels_dir": "valid/labels",
        "test_images_dir": "test/images",
        "test_labels_dir": "test/labels",
        "classes": classes,
    }


def build_model(model_arch: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Instantiate model from SuperGradients models registry."""
    model = models.get(model_arch, num_classes=num_classes, pretrained_weights=None)
    return model.to(device)


def build_dataloaders(dataset_params: dict, batch_size: int, num_workers: int):
    """Create train and validation dataloaders (SuperGradients coco-yolo format)."""
    train_loader = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": dataset_params["train_images_dir"],
            "labels_dir": dataset_params["train_labels_dir"],
            "classes": dataset_params["classes"],
        },
        dataloader_params={"batch_size": batch_size, "num_workers": num_workers},
    )

    val_loader = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": dataset_params["val_images_dir"],
            "labels_dir": dataset_params["val_labels_dir"],
            "classes": dataset_params["classes"],
        },
        dataloader_params={"batch_size": batch_size, "num_workers": num_workers},
    )

    return train_loader, val_loader


def build_train_params(num_epochs: int, num_classes: int) -> dict:
    """Construct training parameters dict for Trainer.train()."""
    return {
        "silent_mode": False,
        "average_best_models": False,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-5,
        "lr_warmup_epochs": 5,
        "initial_lr": 0.01,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.01,
        "optimizer": "SGD",
        "optimizer_params": {"momentum": 0.9, "weight_decay": 1e-4},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.999, "decay_type": "threshold"},
        "max_epochs": num_epochs,
        "mixed_precision": True,
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=num_classes, reg_max=16),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.25,
                top_k_predictions=300,
                num_cls=num_classes,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.2, nms_top_k=300, max_predictions=100, nms_threshold=0.25
                ),
            )
        ],
        "metric_to_watch": "mAP@0.50",
        "greater_metric_to_watch_is_better": True,
    }


def main() -> None:
    """Main entry point: parse args, build data, model, and run training."""
    args = parse_args()

    data_dir = args.data_dir
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        raise SystemExit("No classes provided. Use --classes 'a,b,c'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = args.checkpoint_dir or str(Path.cwd() / "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    dataset_params = prepare_dataset_params(data_dir, classes)
    train_loader, val_loader = build_dataloaders(dataset_params, args.batch_size, args.num_workers)

    model = build_model(args.model_arch, num_classes=len(classes), device=device)

    train_params = build_train_params(num_epochs=args.num_epochs, num_classes=len(classes))

    trainer = Trainer(experiment_name=args.experiment_name, ckpt_root_dir=ckpt_root)

    trainer.train(model=model, training_params=train_params, train_loader=train_loader, valid_loader=val_loader)


if __name__ == "__main__":
    main()
