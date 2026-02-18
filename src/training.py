import argparse
import os
import sys
from datetime import datetime

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from common import read_params
from data_loader import get_data_loader
from metrics import compute_depth_metrics
from model import DPT


class ScaleInvariantLoss(nn.Module):
    """Scale-invariant log loss from Eigen et al., 2014.

    Focuses on relative depth relationships rather than absolute values.
    L = mean(d^2) - lambda * mean(d)^2, where d = log(pred) - log(gt)
    """

    def __init__(self, si_lambda=0.5):
        super().__init__()
        self.si_lambda = si_lambda

    def forward(self, pred, gt):
        valid_mask = gt > 1e-3
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = torch.clamp(pred, min=1e-3)
        d = torch.log(pred) - torch.log(gt)
        return torch.mean(d ** 2) - self.si_lambda * torch.mean(d) ** 2


class GradientLoss(nn.Module):
    """Spatial gradient loss for sharper depth edges.

    Penalizes differences in horizontal/vertical gradients between pred and gt.
    """

    def forward(self, pred, gt):
        valid_mask = gt > 1e-3

        # Horizontal and vertical gradients
        pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        gt_dx = gt[:, :, :, :-1] - gt[:, :, :, 1:]
        gt_dy = gt[:, :, :-1, :] - gt[:, :, 1:, :]

        # Mask for valid gradient regions
        mask_dx = valid_mask[:, :, :, :-1] & valid_mask[:, :, :, 1:]
        mask_dy = valid_mask[:, :, :-1, :] & valid_mask[:, :, 1:, :]

        loss_dx = torch.abs(pred_dx[mask_dx] - gt_dx[mask_dx]).mean() if mask_dx.any() else torch.tensor(0.0, device=pred.device)
        loss_dy = torch.abs(pred_dy[mask_dy] - gt_dy[mask_dy]).mean() if mask_dy.any() else torch.tensor(0.0, device=pred.device)

        return loss_dx + loss_dy


def training(config_path):
    config = read_params(config_path)

    # Data config
    X_dir_train = config["data"]["X_dir_train"]
    y_dir_train = config["data"]["y_dir_train"]
    X_dir_val = config["data"]["X_dir_val"]
    y_dir_val = config["data"]["y_dir_val"]
    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    num_cities = config["data"].get("num_cities", None)
    max_depth = float(config["data"].get("max_depth", 80.0))

    # Model config
    decoder_features = config["model"]["decoder_features"]

    # Training config
    num_epochs = config["training"]["num_epochs"]
    vit_lr = float(config["training"]["vit_lr"])
    decoder_lr = float(config["training"]["decoder_lr"])
    use_amp = config["training"].get("use_amp", False)
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
    warmup_epochs = config["training"].get("warmup_epochs", 2)
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    patience = int(config["training"].get("patience", 10))

    # Loss config
    loss_config = config["training"].get("loss", {})
    si_weight = float(loss_config.get("si_weight", 1.0))
    grad_weight = float(loss_config.get("grad_weight", 0.5))

    # Augmentation config
    aug_config = config.get("augmentation", None)

    # MLflow config
    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"]["run_name"]
    registered_model_name = config["mlflow"]["registered_model_name"]
    server_uri = config["mlflow"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    os.makedirs("saved_models", exist_ok=True)

    train_loader = None
    val_loader = None

    try:
        # Data loaders
        train_loader = get_data_loader(
            X_dir_train, y_dir_train, img_size, batch_size,
            augment=True, aug_config=aug_config, num_cities=num_cities,
            num_workers=num_workers, shuffle=True, pin_memory=True,
        )
        val_loader = get_data_loader(
            X_dir_val, y_dir_val, img_size, batch_size,
            augment=False, num_workers=num_workers, shuffle=False, pin_memory=True,
        )

        # Model
        model = DPT(decoder_channels=decoder_features).to(device)

        optimizer = optim.AdamW(
            [
                {"params": model.vit.parameters(), "lr": vit_lr},
                {"params": model.reassemble_layers.parameters(), "lr": decoder_lr},
                {"params": model.fusion_blocks.parameters(), "lr": decoder_lr},
                {"params": model.output_head.parameters(), "lr": decoder_lr},
            ],
            weight_decay=weight_decay,
        )

        # Loss functions
        si_loss_fn = ScaleInvariantLoss()
        grad_loss_fn = GradientLoss()

        # LR scheduler with warmup
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1))
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

        with mlflow.start_run(run_name=run_name) as mlflow_run:
            mlflow.log_params({
                "batch_size": batch_size,
                "decoder_features": decoder_features,
                "vit_lr": vit_lr,
                "decoder_lr": decoder_lr,
                "num_epochs": num_epochs,
                "use_amp": use_amp,
                "grad_clip_norm": grad_clip_norm,
                "warmup_epochs": warmup_epochs,
                "img_size": img_size,
                "weight_decay": weight_decay,
                "patience": patience,
                "si_weight": si_weight,
                "grad_weight": grad_weight,
                "max_depth": max_depth,
            })

            best_val_rmse = float("inf")
            epochs_without_improvement = 0

            for epoch in range(num_epochs):
                # --- Training ---
                model.train()
                train_loss_sum = 0.0
                total_samples = 0

                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs-1} [Train]", leave=False)
                for i, (images, depths) in enumerate(pbar):
                    images, depths = images.to(device), depths.to(device)
                    optimizer.zero_grad()

                    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                        preds = model(images)
                        loss = si_weight * si_loss_fn(preds, depths) + grad_weight * grad_loss_fn(preds, depths)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_sum += loss.item() * images.size(0)
                    total_samples += images.size(0)
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                train_loss = train_loss_sum / total_samples
                scheduler.step()

                # --- Validation ---
                model.eval()
                all_metrics = []
                with torch.no_grad():
                    vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs-1} [Val]", leave=False)
                    for images, depths in vbar:
                        images, depths = images.to(device), depths.to(device)
                        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                            preds = model(images)
                        metrics = compute_depth_metrics(preds, depths, max_depth=max_depth)
                        all_metrics.append(metrics)
                        vbar.set_postfix({"rmse": f"{metrics['rmse']:.4f}"})

                val_metrics = {
                    k: sum(m[k] for m in all_metrics) / len(all_metrics)
                    for k in all_metrics[0]
                }

                # Log to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                for name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{name}", value, step=epoch)

                current_lr = optimizer.param_groups[0]["lr"]
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

                print(
                    f"Epoch {epoch}/{num_epochs - 1} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val RMSE: {val_metrics['rmse']:.4f} | "
                    f"Val AbsRel: {val_metrics['abs_rel']:.4f} | "
                    f"Val d1: {val_metrics['delta_1']:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

                # Save best model + early stopping
                if val_metrics["rmse"] < best_val_rmse:
                    best_val_rmse = val_metrics["rmse"]
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), "saved_models/model.pth")
                    print(f"  -> New best model saved (val_rmse={best_val_rmse:.4f})")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"  -> Early stopping: no improvement for {patience} epochs")
                        break

            mlflow.pytorch.log_model(
                model,
                f"{mlflow_run.info.run_id}",
                registered_model_name=registered_model_name,
            )

            print(f"\nTraining completed. Best val RMSE: {best_val_rmse:.4f}")

    except KeyboardInterrupt:
        print("\n[Stop] Training interrupted by user. Shutting down workers...")
        mlflow.end_run(status="KILLED")
        # Explicitly exit to trigger finally block and release terminal
        os._exit(0)
    finally:
        print("Cleaning up resources...")
        del train_loader
        del val_loader
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
