import argparse
import os

import numpy as np

from common import read_params

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


def build_engine(onnx_path, engine_path, fp16=True):
    if not TRT_AVAILABLE:
        raise RuntimeError(
            "TensorRT is not installed. Install it from: "
            "https://developer.nvidia.com/tensorrt\n"
            "  pip install tensorrt pycuda"
        )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled for RTX 3060 optimization")

    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"TensorRT engine saved to: {engine_path} ({file_size_mb:.1f} MB)")


def infer_tensorrt(engine_path, image_np):
    """Run inference with a TensorRT engine.

    Args:
        engine_path: path to serialized TensorRT engine
        image_np: numpy array of shape (1, 3, H, W), float32

    Returns:
        numpy array of shape (1, 1, H, W)
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed.")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    input_buf = cuda.mem_alloc(image_np.nbytes)
    output_shape = (1, 1, image_np.shape[2], image_np.shape[3])
    output_np = np.empty(output_shape, dtype=np.float32)
    output_buf = cuda.mem_alloc(output_np.nbytes)

    cuda.memcpy_htod(input_buf, image_np.astype(np.float32))
    context.execute_v2([int(input_buf), int(output_buf)])
    cuda.memcpy_dtoh(output_np, output_buf)

    return output_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 mode")
    args = parser.parse_args()

    config = read_params(args.config)
    onnx_path = config["export"]["onnx_path"]
    engine_path = config["export"]["tensorrt_path"]

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}. Run export_onnx.py first.")
    else:
        build_engine(onnx_path, engine_path, fp16=not args.no_fp16)
