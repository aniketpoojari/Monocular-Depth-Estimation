import os
import sys

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from common import read_params

matplotlib.use("Agg")

CONFIG = read_params("params.yaml")
HF_REPO_ID = os.environ.get("HF_MODEL_REPO", CONFIG["export"]["hf_repo_id"])
HF_FILENAME = "model.onnx"
MODEL_PATH = "saved_models/model.onnx"
IMG_SIZE = CONFIG["data"]["img_size"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SESSION = None


def get_session():
    global SESSION
    if SESSION is None:
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading model from HF Hub: {HF_REPO_ID}/{HF_FILENAME}")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir="saved_models",
            )
        SESSION = ort.InferenceSession(MODEL_PATH)
    return SESSION


def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...]


def colorize_depth(depth, original_size):
    depth = depth.squeeze()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    colored = plt.cm.inferno(depth_norm)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored).resize(original_size, Image.BICUBIC)


def predict(image):
    if image is None:
        return None
    session = get_session()
    original_size = image.size  # (W, H)
    input_tensor = preprocess(image)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    depth_map = outputs[0]
    return colorize_depth(depth_map, original_size)


with gr.Blocks(title="Monocular Depth Estimation") as demo:
    gr.Markdown(
        "# Monocular Depth Estimation\n"
        "Upload an image to predict its depth map using a "
        "Dense Prediction Transformer (DPT) model trained on Cityscapes. "
        "The model uses a ViT-Base/16 encoder with convolutional decoder blocks."
    )
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        output_depth = gr.Image(type="pil", label="Predicted Depth Map")

    btn = gr.Button("Predict Depth", variant="primary")
    btn.click(fn=predict, inputs=input_image, outputs=output_depth)

    gr.Examples(
        examples=["examples/example1.png", "examples/example2.png"],
        inputs=input_image,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
