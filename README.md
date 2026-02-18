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
   (Conv + BatchNorm + ReLU + bilinear upsample)
        |
   Output head (Conv -> Upsample -> Conv -> ReLU)
        |
   Depth Map (224x224x1) in meters
```

The encoder extracts multi-scale features from intermediate ViT layers. Reassemble layers project and resample these features to different spatial resolutions using bilinear interpolation + Conv2d (avoids checkerboard artifacts from ConvTranspose2d). Fusion blocks progressively combine features via skip connections and upsampling. The output head produces the final single-channel metric depth prediction in meters.

## Results

*Results on Cityscapes test set (metric depth in meters, max_depth=80m):*

| Metric | Value | Direction |
|--------|-------|-----------|
| AbsRel | 0.1536 | lower is better |
| SqRel  | 2.1926 | lower is better |
| RMSE   | 10.19m | lower is better |
| RMSE log | 0.2399 | lower is better |
| delta < 1.25 | 76.47% | higher is better |
| delta < 1.25^2 | 94.03% | higher is better |
| delta < 1.25^3 | 97.83% | higher is better |

### Understanding the Metrics

These are the **standard monocular depth estimation metrics** established by [Eigen et al., 2014](https://arxiv.org/abs/1406.2283), used across all major depth estimation benchmarks (KITTI, NYU Depth V2, Cityscapes).

**Error metrics** (lower is better):

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **AbsRel** | mean(\|pred - gt\| / gt) | Average relative error — how far off predictions are as a percentage of ground truth. The primary ranking metric. |
| **SqRel** | mean((pred - gt)^2 / gt) | Squared relative error — penalizes large errors more heavily than AbsRel. |
| **RMSE** | sqrt(mean((pred - gt)^2)) | Root mean squared error in meters — measures absolute accuracy. |
| **RMSE log** | sqrt(mean((log(pred) - log(gt))^2)) | RMSE in log-space — scale-invariant, errors at 1m and 10m are weighted equally. |

**Accuracy metrics** (higher is better):

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **delta < 1.25** | % of pixels where max(pred/gt, gt/pred) < 1.25 | Fraction of pixels within 25% of ground truth. |
| **delta < 1.25^2** | Same with threshold 1.5625 | Fraction within ~56%. |
| **delta < 1.25^3** | Same with threshold ~1.953 | Fraction within ~95%. |

### How Benchmarking Works

Evaluation uses a held-out **test set** that the model never sees during training or validation:

```
Training:    Train split --> model learns weights
Validation:  Val split   --> monitor overfitting, select best checkpoint (by val RMSE)
Testing:     Test split  --> final benchmark (all 7 metrics), reported in results above
```

1. The best model checkpoint (lowest validation RMSE) is selected during training
2. `src/evaluate.py` loads this checkpoint and runs inference on every test image
3. All 7 metrics are computed on valid pixels (depth > 0.001m) and averaged
4. Test metrics are saved to `metrics.json` and logged to the MLflow training run

## Project Structure

```
├── src/
│   ├── model.py              # DPT architecture (ViT encoder + decoder)
│   ├── data_loader.py         # Dataset with augmentation and normalization
│   ├── training.py            # Training loop with AMP, validation, early stopping
│   ├── evaluate.py            # Test set evaluation, logs metrics to MLflow
│   ├── log_production_model.py # Conditional model promotion (only if improved)
│   ├── export_onnx.py         # Export production model to ONNX format
│   ├── push_to_hf.py          # Upload ONNX model to Hugging Face Hub
│   ├── export_tensorrt.py     # Convert ONNX to TensorRT (GPU optimization)
│   ├── metrics.py             # Standard depth estimation metrics
│   ├── preprocess.py          # Disparity → metric depth conversion + resize
│   └── common.py              # Config utilities
├── tests/
│   ├── test_model.py          # Model architecture tests
│   ├── test_data_loader.py    # Data pipeline tests
│   └── test_metrics.py        # Metrics correctness tests
├── .github/
│   └── workflows/
│       └── deploy-hf-spaces.yml  # Auto-deploy to HF Spaces on push
├── app.py                     # Gradio web demo (ONNX Runtime inference)
├── Dockerfile                 # HF Spaces Docker deployment
├── params.yaml                # All hyperparameters and paths
├── dvc.yaml                   # Reproducible ML pipeline (6 stages)
├── dvc.lock                   # Pipeline state tracking
├── requirements.txt           # Dependencies
└── system_design.md           # Architecture and design decisions
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
data/raw/depth_y/{train,val,test}/{city_name}/*.png   # Disparity maps (uint16)
```

## Pipeline

The project uses [DVC](https://dvc.org/) to manage a reproducible ML pipeline with 6 stages:

```
preprocessing → training → evaluate → log_production_model → export_onnx → push_to_hf
```

| Stage | Script | Description |
|-------|--------|-------------|
| `preprocessing` | `src/preprocess.py` | Resizes images, converts disparity to metric depth (.npy) |
| `training` | `src/training.py` | Trains DPT with SI + gradient loss, logs to MLflow |
| `evaluate` | `src/evaluate.py` | Evaluates on test set, logs metrics to MLflow |
| `log_production_model` | `src/log_production_model.py` | Promotes model only if test RMSE improved (via MLflow) |
| `export_onnx` | `src/export_onnx.py` | Exports production model to ONNX with verification |
| `push_to_hf` | `src/push_to_hf.py` | Uploads ONNX model to Hugging Face Hub |

Run the full pipeline:

```bash
dvc repro
```

Run a single stage:

```bash
dvc repro evaluate --single-item
```

## Training

Configure hyperparameters in `params.yaml`, then run:

```bash
dvc repro
```

Key training features:
- **Metric depth output** — Cityscapes disparity converted to meters via `depth = baseline × focal_length / (disparity / 256)`, clamped to 80m
- **Scale-invariant + gradient loss** — SI loss for relative depth accuracy, gradient loss for sharp edges
- **AdamW optimizer** with decoupled weight decay (0.01)
- **Differential learning rates** — 1e-5 for pretrained ViT, 1e-4 for decoder
- **Mixed precision (AMP)** for faster training on CUDA GPUs
- **Cosine LR scheduler** with linear warmup (3 epochs)
- **Early stopping** (patience=10) to prevent overfitting
- **Gradient clipping** (max norm=1.0) for training stability
- **Data augmentation**: horizontal flip, color jitter, random resized crop
- **ImageNet normalization** for ViT transfer learning

### MLflow Tracking

Training and test metrics are logged to MLflow with a local SQLite backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each run tracks:
- Training parameters (batch size, LRs, epochs, loss weights, etc.)
- Per-epoch metrics: train_loss, val_rmse, val_abs_rel, val_delta_1, learning_rate
- Test metrics: test_rmse, test_abs_rel, test_delta_1, etc. (logged by evaluate stage)
- Full PyTorch model artifact

## Evaluation

```bash
dvc repro evaluate --single-item
```

Evaluates the best model on the test set, writes `metrics.json`, and logs test metrics to the latest MLflow run. View metrics history:

```bash
dvc metrics show
dvc metrics diff
```

## Model Export

### ONNX

```bash
python src/export_onnx.py --config=params.yaml
```

Exports `saved_models/production_model.pth` to `saved_models/model.onnx` with ONNX Runtime verification (opset 17).

### TensorRT (optional, requires NVIDIA TensorRT SDK)

```bash
python src/export_tensorrt.py --config=params.yaml
```

Builds an FP16-optimized TensorRT engine for GPU deployment.

## Demo

Run the Gradio web interface locally:

```bash
python app.py
```

The app downloads the ONNX model from Hugging Face Hub if not available locally. Uses ONNX Runtime for inference — no GPU required.

### Hugging Face Spaces

The demo is auto-deployed to [HF Spaces](https://huggingface.co/spaces/aniketp2009gmail/depth-estimation-demo) via GitHub Actions on push to `main`. The Space uses a Docker container with ONNX Runtime for CPU inference.

## Tests

```bash
pytest tests/ -v
```

| Test File | What It Covers |
|-----------|---------------|
| `test_model.py` | Forward pass shapes, CPU compatibility, output non-negativity, decoder channel variants |
| `test_data_loader.py` | Dataset length, item shapes, batching, augmentation consistency |
| `test_metrics.py` | Perfect prediction correctness, metric keys, empty mask handling |

## CI/CD

- **GitHub Actions** (`deploy-hf-spaces.yml`): auto-deploys Gradio app to HF Spaces on push to `main`
- **DVC Pipeline**: `push_to_hf` stage uploads ONNX model to HF Hub after successful training + evaluation

## References

- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413). ICCV 2021.
- Eigen, D., Puhrsch, C., & Fergus, R. (2014). [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283). NeurIPS 2014.
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [timm - PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
