import argparse
import os
import shutil

import mlflow
import pandas as pd

from common import read_params


def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["server_uri"])

    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        print("MLflow experiment not found.")
        return

    df = pd.DataFrame(mlflow.search_runs(experiment_ids=experiment.experiment_id))
    df = df[df["status"] == "FINISHED"]

    # Only consider runs that have been evaluated (test_rmse logged)
    df = df.dropna(subset=["metrics.test_rmse"])
    if df.empty:
        print("No evaluated runs found — saving current model as production model.")
        shutil.copy2("saved_models/model.pth", "saved_models/production_model.pth")
        print("Production model saved: saved_models/production_model.pth")
        return

    # Find the best run (lowest test_rmse) and the latest run
    best_run = df.loc[df["metrics.test_rmse"].idxmin()]
    latest_run = df.sort_values("start_time", ascending=False).iloc[0]

    best_rmse = best_run["metrics.test_rmse"]
    latest_rmse = latest_run["metrics.test_rmse"]
    latest_run_id = latest_run["run_id"]

    print(f"Latest run:  {latest_run_id} (test_rmse={latest_rmse:.4f})")
    print(f"Best run:    {best_run['run_id']} (test_rmse={best_rmse:.4f})")

    if latest_run_id == best_run["run_id"]:
        # Latest run is the best — promote to production
        shutil.copy2("saved_models/model.pth", "saved_models/production_model.pth")
        print(f"Model improved! Production model updated (test_rmse={latest_rmse:.4f})")
    else:
        print(f"No improvement (best={best_rmse:.4f}, current={latest_rmse:.4f})")
        print("Production model NOT updated.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
