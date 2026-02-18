# Monocular Depth Estimation with Dense Prediction Transformer

Monocular depth estimation from single RGB images using a **Vision Transformer (ViT-Base/16)** encoder with convolutional decoder blocks, trained on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.

Based on the DPT architecture: [Vision Transformers for Dense Prediction (Ranftl et al., 2021)](https://arxiv.org/abs/2103.13413)

## Architecture

```
Input Image (224x224x3)
        |
   ViT-Base/16 Encoder (pretrained on ImageNet)
        |
   Extract tokens at layers 3, 6, 9, 12
        |
   +----+----+--------+--------+
   | Reassemble layers at 4 scales  |
   | (1/4, 1/8, 1/16, 1/32)        |
   +----+----+--------+--------+
        |
   Fusion blocks with skip connections
        |
   Output head (Conv -> Upsample -> Conv -> ReLU)
        |
   Depth Map (224x224x1)
```

The encoder extracts multi-scale features from intermediate ViT layers. Reassemble layers project and resample these features to different spatial resolutions. Fusion blocks progressively combine features via skip connections and upsampling. The output head produces the final single-channel depth prediction.

## Results

*Results on Cityscapes test set:*

| Metric | Value | Direction |
|--------|-------|-----------|
| AbsRel | 0.1895 | lower is better |
| SqRel  | 0.0171 | lower is better |
| RMSE   | 0.0796 | lower is better |
| RMSE log | 0.3141 | lower is better |
| delta < 1.25 | 0.7339 | higher is better |
| delta < 1.25^2 | 0.8940 | higher is better |
| delta < 1.25^3 | 0.9539 | higher is better |

### Understanding the Metrics

These are the **standard monocular depth estimation metrics** established by [Eigen et al., 2014](https://arxiv.org/abs/1406.2283), used across all major depth estimation benchmarks (KITTI, NYU Depth V2, Cityscapes). Every depth estimation paper reports these 7 metrics to enable fair comparison.

**Error metrics** (lower is better):

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **AbsRel** | mean(\|pred - gt\| / gt) | Average relative error — how far off predictions are as a percentage of ground truth. The primary ranking metric in most papers. |
| **SqRel** | mean((pred - gt)^2 / gt) | Squared relative error — penalizes large errors more heavily than AbsRel, useful for detecting outliers. |
| **RMSE** | sqrt(mean((pred - gt)^2)) | Root mean squared error in depth units — measures absolute accuracy. Sensitive to large errors. |
| **RMSE log** | sqrt(mean((log(pred) - log(gt))^2)) | RMSE in log-space — scale-invariant, so errors at 1m and 10m depth are weighted equally. |

**Accuracy metrics** (higher is better):

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **delta < 1.25** | % of pixels where max(pred/gt, gt/pred) < 1.25 | Fraction of pixels within 25% of ground truth. The strictest accuracy threshold. |
| **delta < 1.25^2** | Same with threshold 1.5625 | Fraction within ~56% — moderate accuracy. |
| **delta < 1.25^3** | Same with threshold ~1.953 | Fraction within ~95% — relaxed accuracy, almost all reasonable predictions pass. |

**Why all 7?** No single metric tells the full story. A model could have low RMSE but poor AbsRel (biased toward large depths), or high delta_1 but bad RMSE_log (most pixels correct but a few wildly off in scale). Reporting all 7 enables direct comparison with published results.

### How Benchmarking Works

Evaluation uses a held-out **test set** that the model never sees during training or validation:

```
Training:    Train split --> model learns weights
Validation:  Val split   --> monitor overfitting, select best checkpoint (by val RMSE)
Testing:     Test split  --> final benchmark (all 7 metrics), reported in results above
```

1. The best model checkpoint (lowest validation RMSE) is selected from MLflow
2. `src/evaluate.py` loads this checkpoint and runs inference on every test image
3. For each batch, all 7 metrics are computed on valid pixels (depth > 0.001) via `src/metrics.py`
4. Metrics are averaged across all batches and saved to `metrics.json`

Only the test metrics are reported as results — validation metrics are used solely for model selection during training.

## Project Structure

```
├── src/
│   ├── model.py              # DPT architecture (ViT encoder + decoder)
│   ├── data_loader.py         # Dataset with augmentation and normalization
│   ├── training.py            # Training loop with AMP, validation, LR scheduler
│   ├── evaluate.py            # Test set evaluation with standard metrics
│   ├── inference.py           # Visual inference with colorized depth maps
│   ├── export_onnx.py         # Export model to ONNX format
│   ├── export_tensorrt.py     # Convert ONNX to TensorRT (GPU optimization)
│   ├── log_production_model.py # Select best model from MLflow
│   ├── metrics.py             # Depth estimation metric functions
│   ├── preprocess.py          # Resize raw images to target resolution
│   └── common.py              # Config utilities
├── tests/
│   ├── test_model.py          # Model architecture tests
│   ├── test_data_loader.py    # Data pipeline tests
│   └── test_metrics.py        # Metrics correctness tests
├── app.py                     # Gradio web demo (ONNX Runtime inference)
├── params.yaml                # All hyperparameters and paths
├── dvc.yaml                   # Reproducible ML pipeline (4 stages)
├── dvc.lock                   # Pipeline state tracking
├── requirements.txt           # Dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (for training)
- [Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/) (leftImg8bit + disparity)

### Installation

```bash
git clone https://github.com/aniketpoojari/Monocular-Depth-Estimation.git
cd Monocular-Depth-Estimation
pip install -r requirements.txt
```

### Data

Download from Cityscapes and organize as:
```
data/raw/depth_X/{train,val,test}/{city_name}/*.png   # RGB images
data/raw/depth_y/{train,val,test}/{city_name}/*.png   # Disparity maps
```

## Pipeline

The project uses [DVC](https://dvc.org/) to manage a reproducible ML pipeline with 4 stages:

```
preprocessing -> training -> evaluate -> log_production_model
```

| Stage | Script | Description |
|-------|--------|-------------|
| `preprocessing` | `src/preprocess.py` | Resizes raw images to 224x224 |
| `training` | `src/training.py` | Trains the DPT model with AMP, logs to MLflow |
| `evaluate` | `src/evaluate.py` | Evaluates on test set, writes `metrics.json` |
| `log_production_model` | `src/log_production_model.py` | Selects best model from MLflow by val RMSE |

Run the full pipeline:

```bash
dvc repro
```

Run a single stage:

```bash
dvc repro training
```

## Training

Configure hyperparameters in `params.yaml`, then run:

```bash
dvc repro
```

Key training features:
- **Metric depth output** — disparity converted to meters via camera intrinsics (`depth = baseline × focal_length / disparity`), clamped to 80m
- **Scale-invariant + gradient loss** for relative depth accuracy and sharp edges
- **AdamW optimizer** with decoupled weight decay (0.01)
- **Mixed precision (AMP)** for faster training on CUDA GPUs
- **Cosine LR scheduler** with linear warmup
- **Early stopping** (patience=10) to prevent overfitting
- **Gradient clipping** for training stability
- **Data augmentation**: random horizontal flip, color jitter, random resized crop
- **ImageNet normalization** for ViT transfer learning
- **Validation metrics** logged per epoch to MLflow

### MLflow Tracking

Training metrics are logged to MLflow with a local SQLite backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Evaluation

```bash
python src/evaluate.py --config=params.yaml
```

Evaluates the best model on the test set and outputs `metrics.json` with all 7 standard depth metrics. See the [Results](#results) section for metric definitions and how benchmarking works.

## Inference

```bash
# Single image
python src/inference.py --image path/to/image.png

# With ground truth comparison
python src/inference.py --image path/to/image.png --gt path/to/depth.png

# Custom output directory
python src/inference.py --image path/to/image.png --output-dir results/
```

## Model Export

### ONNX

```bash
python src/export_onnx.py --config=params.yaml
```

Exports to `saved_models/model.onnx` with ONNX Runtime verification.

### TensorRT (optional, requires NVIDIA TensorRT SDK)

```bash
python src/export_tensorrt.py --config=params.yaml
```

Builds an FP16-optimized TensorRT engine for GPU deployment.

## Demo

Run the Gradio web interface locally:

```bash
# Export ONNX model first
python src/export_onnx.py --config=params.yaml

# Launch demo
python app.py
```

The app uses ONNX Runtime for inference, so no GPU is required to run the demo.

## Tests

```bash
python -m pytest tests/ -v
```

| Test File | What It Covers |
|-----------|---------------|
| `test_model.py` | Forward pass shapes, CPU compatibility, output non-negativity, decoder channel variants |
| `test_data_loader.py` | Dataset length, item shapes, batching, augmentation consistency |
| `test_metrics.py` | Perfect prediction correctness, metric keys, empty mask handling |

## References

- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413). ICCV 2021.
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [timm - PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
