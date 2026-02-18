import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import read_params
from model import DPT


def export_onnx(config_path):
    config = read_params(config_path)

    img_size = config["data"]["img_size"]
    decoder_features = config["model"]["decoder_features"]
    onnx_path = config["export"]["onnx_path"]
    opset = config["export"]["opset_version"]

    device = torch.device("cpu")

    model = DPT(decoder_channels=decoder_features).to(device)
    state = torch.load("saved_models/production_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=opset,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes={"image": {0: "batch"}, "depth": {0: "batch"}},
    )
    print(f"ONNX model exported to: {onnx_path}")

    # Verify model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")

    # Compare outputs
    session = ort.InferenceSession(onnx_path)
    ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = session.run(None, ort_input)[0]

    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    max_diff = np.abs(pt_output - ort_output).max()
    print(f"Max difference (PyTorch vs ONNX Runtime): {max_diff:.6f}")

    if max_diff < 1e-3:
        print("Verification PASSED.")
    else:
        print(f"WARNING: Output difference ({max_diff:.6f}) exceeds threshold.")

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Model size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DPT model to ONNX format")
    parser.add_argument("--config", default="params.yaml")
    export_onnx(parser.parse_args().config)
