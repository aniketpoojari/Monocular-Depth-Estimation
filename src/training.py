from common import read_params
from data_loader import get_data_loader
import argparse
from model import DPT
import torch.optim as optim
import torch.nn as nn
import torch
import mlflow
from datetime import datetime


def training(config_path):
    config = read_params(config_path)
    X_dir_train = config["data"]["X_dir_train"]
    y_dir_train = config["data"]["y_dir_train"]
    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]

    decoder_features = config["model"]["decoder_features"]

    num_epoches = config["training"]["num_epoches"]
    vit_lr = config["training"]["vit_lr"]
    decoder_lr = config["training"]["decoder_lr"]

    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"]["run_name"]
    registered_model_name = config["mlflow"]["registered_model_name"]
    server_uri = config["mlflow"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    train_dataloader = get_data_loader(X_dir_train, y_dir_train, img_size, batch_size)

    model = DPT(new=256).cuda()  # For example, 21 classes for segmentation
    opt = optim.Adam(
        [
            {"params": model.vit.parameters(), "lr": 1e-5},  # Encoder/backbone
            {
                "params": model.reassemble_layers.parameters(),
                "lr": 1e-4,
            },  # Decoder's reassemble layers
            {
                "params": model.fusion_blocks.parameters(),
                "lr": 1e-4,
            },  # Decoder's fusion blocks
            {
                "params": model.output_head.parameters(),
                "lr": 1e-4,
            },  # Decoder's output head
        ]
    )

    cri = nn.MSELoss().cuda()

    with mlflow.start_run(run_name=run_name) as mlflow_run:

        mlflow.log_params(
            {
                "batch_size": batch_size,
                "decoder_features": decoder_features,
                "vit_lr": vit_lr,
                "decoder_lr": decoder_lr,
                "num_epoches": num_epoches,
            }
        )

        for epoch in range(num_epoches):
            running_loss = []
            s = 0
            for X, y in train_dataloader:
                inputs, labels = X.cuda(), y.cuda()
                opt.zero_grad()
                outputs = model(inputs)
                loss = torch.sqrt(cri(outputs, labels))
                loss.backward()
                opt.step()
                running_loss.append(loss.item() * X.shape[0])
                s += X.shape[0]
            loss = sum(running_loss) / s

            mlflow.log_metric("Train Loss - RMSE - cm", loss, step=epoch)
            print("Epoch: {} Loss: {}".format(epoch, loss))

        mlflow.pytorch.log_model(
            model,
            f"{mlflow_run.info.run_id}",
            registered_model_name=registered_model_name,
        )

        with open("training_completion.txt", "w") as file:
            # Get the current date and time
            current_datetime = datetime.now()
            # Format the date and time as a string
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
