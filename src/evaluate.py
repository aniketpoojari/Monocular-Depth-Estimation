import argparse
import json
import os

import mlflow
import pandas as pd
import torch

from common import read_params
from data_loader import get_data_loader
from metrics import compute_depth_metrics
from model import DPT


def evaluate(config_path):
    config = read_params(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    max_depth = float(config["data"].get("max_depth", 80.0))
    decoder_features = config["model"]["decoder_features"]

    test_loader = get_data_loader(
        config["data"]["X_dir_test"],
        config["data"]["y_dir_test"],
        img_size,
        batch_size,
        augment=False,
        num_workers=num_workers,
        shuffle=False,
    )

    model = DPT(decoder_channels=decoder_features).to(device)
    state = torch.load("saved_models/model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_metrics = []
    with torch.no_grad():
        for images, depths in test_loader:
            images, depths = images.to(device), depths.to(device)
            preds = model(images)
            metrics = compute_depth_metrics(preds, depths, max_depth=max_depth)
            all_metrics.append(metrics)

    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics)
        for k in all_metrics[0]
    }

    # Save metrics.json for DVC tracking
    with open("metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    print("\nTest Set Metrics:")
    print("-" * 40)
    for name, value in avg_metrics.items():
        print(f"  {name:>10s}: {value:.4f}")
    print("-" * 40)
    print(f"Saved to metrics.json")

    # Log test metrics to the latest MLflow training run
    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["server_uri"])

    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        print("MLflow experiment not found — skipping metric logging.")
        return

    df = pd.DataFrame(mlflow.search_runs(experiment_ids=experiment.experiment_id))
    df = df[df["status"] == "FINISHED"].sort_values("start_time", ascending=False)

    if df.empty:
        print("No finished MLflow runs found — skipping metric logging.")
        return

    latest_run_id = df["run_id"].values[0]
    with mlflow.start_run(run_id=latest_run_id):
        for name, value in avg_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

    print(f"Logged test metrics to MLflow run: {latest_run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    evaluate(parser.parse_args().config)
