from common import read_params
import argparse
import mlflow
import pandas as pd
import shutil


def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow"]
    experiment_name = mlflow_config["experiment_name"]
    server_uri = mlflow_config["server_uri"]
    mlflow.set_tracking_uri(server_uri)

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    df = pd.DataFrame(mlflow.search_runs(experiment_ids=experiment_id))
    df = df[df["status"] == "FINISHED"]
    df = df[
        df["metrics.Train Loss - RMSE - cm"]
        == df["metrics.Train Loss - RMSE - cm"].min()
    ]
    run_id = df["run_id"].values
    artifact_uri = df["artifact_uri"].values
    src = f"{artifact_uri[0]}/{run_id[0]}/data/model.pth"
    dest = "saved_models/model.pth"
    shutil.copyfile(src, dest)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
