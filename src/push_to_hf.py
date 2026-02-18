import argparse
import os

from huggingface_hub import HfApi

from common import read_params


def push_to_hf(config_path):
    config = read_params(config_path)

    onnx_path = config["export"]["onnx_path"]
    repo_id = config["export"]["hf_repo_id"]
    token = os.environ.get("HF_TOKEN")

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}. Skipping upload.")
        return

    api = HfApi(token=token)

    # Create model repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Model repo ready: https://huggingface.co/{repo_id}")

    api.upload_file(
        path_or_fileobj=onnx_path,
        path_in_repo="model.onnx",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Uploaded {onnx_path} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    push_to_hf(parser.parse_args().config)
