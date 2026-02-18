# System Design: Monocular Depth Estimation

## 1. Overview

This system estimates per-pixel depth from a single RGB image using a Dense Prediction Transformer (DPT). It covers the full ML lifecycle: data preprocessing, model training, experiment tracking, evaluation, conditional model promotion, ONNX export, Hugging Face deployment, and serving via a web demo.

## 2. System Architecture

```
+---------------------+       +---------------------+       +---------------------+
|    Data Ingestion    |       |   Training Pipeline |       |     Serving Layer   |
|                     |       |                     |       |                     |
| Cityscapes Dataset  +------>+ DVC Pipeline        +------>+ Gradio App (ONNX)   |
| (leftImg8bit +      |       | (preprocess, train, |       | on HF Spaces        |
|  disparity maps)    |       |  evaluate, promote, |       | (Docker + CPU)      |
|                     |       |  export, push)      |       |                     |
+---------------------+       +----------+----------+       +---------------------+
                                         |
                              +----------+----------+
                              |                     |
                       +------v------+       +------v------+
                       | MLflow      |       | HF Hub      |
                       | (SQLite)    |       | (model.onnx)|
                       +-------------+       +-------------+
```

## 3. Data Flow

### 3.1 Raw Data

```
data/raw/
├── depth_X/{train,val,test}/{city}/*.png    # RGB images (variable resolution)
└── depth_y/{train,val,test}/{city}/*.png    # Disparity maps (uint16, disparity × 256)
```

Source: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with leftImg8bit (RGB) and disparity packages.

### 3.2 Preprocessing

**Script**: `src/preprocess.py`

```
Raw RGB (1024x2048)       --> Resize (BICUBIC)  --> Processed PNG (224x224)
Raw disparity (uint16)    --> Resize (NEAREST)   --> Convert to metric depth --> .npy (float32)
```

Disparity to depth conversion:
```
depth_meters = (baseline × focal_length) / (raw_value / 256)
```
- Cityscapes stores `disparity × 256` as uint16
- Camera intrinsics: baseline=0.209313m, focal_length=2262.52px
- Invalid pixels (disparity=0) kept as 0
- Clamped to [0, max_depth] (default 80m)
- Saved as `.npy` float32 arrays — avoids runtime conversion during training

### 3.3 Data Loading

**Script**: `src/data_loader.py`

```
Preprocessed .npy depth + RGB PNG --> Augmentation --> Tensor --> Normalize --> Batch
```

- Loads pre-converted `.npy` metric depth files directly (no runtime disparity conversion)
- **Image normalization**: ImageNet mean/std (required for pretrained ViT)
- **Augmentations** (train only): horizontal flip, color jitter, random resized crop
- Augmentations applied consistently to image-depth pairs (same flip, same crop)
- `persistent_workers=True` keeps DataLoader workers alive between epochs

## 4. Model Architecture

**Script**: `src/model.py`

### 4.1 DPT (Dense Prediction Transformer)

Based on [Ranftl et al., 2021](https://arxiv.org/abs/2103.13413).

```
                    Input (B, 3, 224, 224)
                           |
                    ViT-Base/16 Encoder
                    (pretrained, 12 blocks)
                           |
              Extract tokens at blocks 3, 6, 9, 12
                           |
            +------+-------+-------+------+
            |      |       |       |      |
        Reassemble Reassemble Reassemble Reassemble
        (scale=4)  (scale=8) (scale=16) (scale=32)
            |      |       |       |
            v      v       v       v
        (56x56) (28x28) (14x14)  (7x7)
            |      |       |       |
            +------+-------+-------+
                           |
                   Fusion Blocks (x4)
           (Conv + BatchNorm + ReLU + bilinear upsample)
                           |
                    Output Head
              (Conv -> Upsample -> Conv -> ReLU)
                           |
                  Depth Map (B, 1, 224, 224)
                  (metric depth in meters)
```

### 4.2 Components

| Component | Description | Parameters |
|-----------|-------------|------------|
| **ViT-Base/16** | Pretrained encoder from timm. Patch size 16, embed dim 768, 12 transformer blocks | ~86M |
| **ReassembleLayer** (x4) | Projects 768-dim tokens to decoder channels, resamples via bilinear interpolation + Conv2d | ~3M |
| **FusionBlock** (x4) | Skip connection + Conv + BatchNorm + ReLU, then bilinear upsample + Conv2d | ~1M |
| **Output Head** | 3-layer conv head: channels -> channels/2 -> 32 -> 1, with bilinear upsample and ReLU | ~0.1M |

**Total**: ~90M parameters, ~358 MB (ONNX)

### 4.3 Design Decisions

- **Why ViT over CNN encoder?** ViT captures global context in every layer via self-attention, critical for depth estimation where distant objects inform relative scale.
- **Why extract at layers 3, 6, 9, 12?** Provides multi-scale representations — early layers capture local texture, later layers capture global structure.
- **Why ReLU on output?** The model predicts metric depth in meters (0 to 80m), which is non-negative and unbounded above. ReLU enforces non-negativity without the gradient saturation of Sigmoid. Previously, Sigmoid was used when depth was normalized to [0, 1], but this caused vanishing gradients at depth extremes.
- **Why scale-invariant + gradient loss?** (1) **Scale-invariant log loss** (Eigen et al., 2014) focuses on relative depth — a 1m error at 5m is penalized more than at 50m. Formula: `mean(d²) - λ·mean(d)²` where `d = log(pred) - log(gt)`. (2) **Gradient loss** penalizes spatial gradient differences, producing sharper edges. Total: `si_weight × SI_loss + grad_weight × grad_loss`.
- **Why bilinear upsample + Conv2d instead of ConvTranspose2d?** ConvTranspose2d produces checkerboard artifacts when kernel size isn't divisible by stride. Bilinear interpolation + Conv2d produces smooth upsampling without artifacts.
- **Why BatchNorm + ReLU in FusionBlock?** BatchNorm stabilizes decoder training across varying feature magnitudes. ReLU adds non-linearity for blending skip connection features.

## 5. Training Pipeline

**Script**: `src/training.py`

### 5.1 Training Loop

```
For each epoch:
    1. Train pass (forward + backward + optimizer step)
    2. Validation pass (forward only, compute metrics)
    3. Log metrics to MLflow
    4. Save model if val RMSE improves
    5. Check early stopping (patience epochs without improvement)
    6. Step LR scheduler
```

### 5.2 Optimization Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Optimizer | AdamW | Decoupled weight decay, standard for transformers |
| Weight decay | 0.01 | Regularizes decoder; ViT benefits from decoupled decay |
| ViT LR | 1e-5 | Low LR for pretrained encoder |
| Decoder LR | 1e-4 | Higher LR for randomly initialized decoder |
| Scheduler | Cosine + linear warmup (3 epochs) | Warmup stabilizes early training; cosine avoids sudden drops |
| AMP | Enabled | FP16 reduces memory and increases throughput |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Loss | SI log + gradient | SI captures relative depth; gradient sharpens edges |
| Early stopping | patience=10 | Stops if val RMSE doesn't improve for 10 epochs |

### 5.3 Differential Learning Rates

```python
optimizer = AdamW([
    {"params": model.vit.parameters(),              "lr": 1e-5},   # pretrained
    {"params": model.reassemble_layers.parameters(), "lr": 1e-4},  # random init
    {"params": model.fusion_blocks.parameters(),     "lr": 1e-4},  # random init
    {"params": model.output_head.parameters(),       "lr": 1e-4},  # random init
], weight_decay=0.01)
```

## 6. Evaluation

**Script**: `src/evaluate.py`

### 6.1 Metrics

All metrics computed in `src/metrics.py` on valid pixels (depth > 0.001m, clamped to max_depth=80m):

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| AbsRel | mean(\|pred - gt\| / gt) | Relative accuracy |
| SqRel | mean((pred - gt)^2 / gt) | Penalizes large relative errors |
| RMSE | sqrt(mean((pred - gt)^2)) | Absolute accuracy (meters) |
| RMSE log | sqrt(mean((log(pred) - log(gt))^2)) | Scale-invariant accuracy |
| delta < 1.25 | % pixels where max(pred/gt, gt/pred) < 1.25 | Inlier ratio (strict) |
| delta < 1.25^2 | Same with threshold 1.5625 | Inlier ratio (moderate) |
| delta < 1.25^3 | Same with threshold 1.953125 | Inlier ratio (relaxed) |

### 6.2 MLflow Integration

After computing test metrics, `evaluate.py` logs them to the latest MLflow training run as `test_rmse`, `test_abs_rel`, etc. This enables `log_production_model.py` to compare runs via MLflow rather than local files.

## 7. Experiment Tracking

**Tool**: MLflow with SQLite backend

### 7.1 What Gets Tracked

- **Parameters**: batch size, learning rates, epochs, image size, AMP, grad clip, weight decay, patience, loss weights, max depth
- **Training metrics per epoch**: train_loss, val_rmse, val_abs_rel, val_delta_1, learning_rate
- **Test metrics** (from evaluate stage): test_rmse, test_abs_rel, test_delta_1, etc.
- **Artifacts**: Full PyTorch model via `mlflow.pytorch.log_model`
- **Model Registry**: Best model registered as `ViT+Decoder` with versioning

### 7.2 Production Model Selection

**Script**: `src/log_production_model.py`

Conditional model promotion using MLflow as source of truth:

1. Query all finished MLflow runs that have `test_rmse` logged
2. Find the run with the lowest `test_rmse` (best overall)
3. If the latest run is the best → copy `model.pth` to `production_model.pth`
4. If not → keep existing `production_model.pth` unchanged

This ensures only improved models are promoted to production and exported.

## 8. Model Export

### 8.1 ONNX Export

**Script**: `src/export_onnx.py`

```
production_model.pth --> torch.onnx.export --> model.onnx --> Verification
```

- Loads from `saved_models/production_model.pth` (only promoted models)
- Opset version 17
- Dynamic batch axis
- Verified by comparing PyTorch and ONNX Runtime outputs (threshold: max diff < 1e-3)
- Output: `saved_models/model.onnx` (~358 MB)

### 8.2 Hugging Face Hub Upload

**Script**: `src/push_to_hf.py`

- Creates model repo on HF Hub if it doesn't exist
- Uploads `model.onnx` to the repo
- Uses `HF_TOKEN` environment variable for authentication

### 8.3 TensorRT Export (Optional)

**Script**: `src/export_tensorrt.py`

- FP16 mode for GPU-optimized inference
- Requires NVIDIA TensorRT SDK and pycuda

## 9. Serving

### 9.1 Gradio Web App

**Script**: `app.py`

```
User uploads image
        |
   Preprocess (resize 224x224, normalize with ImageNet stats)
        |
   ONNX Runtime inference (CPU)
        |
   Colorize depth map (inferno colormap)
        |
   Resize to original dimensions
        |
   Display side-by-side (input + depth)
```

- Downloads ONNX model from HF Hub if not available locally
- Uses ONNX Runtime (CPU-compatible, no GPU required)
- Lazy-loads the ONNX session on first request
- Reads `img_size` from `params.yaml` (consistent with training)

### 9.2 Deployment

| Target | Method | Notes |
|--------|--------|-------|
| **HF Spaces** | Docker + GitHub Actions | Auto-deploys on push to main. Model downloaded from HF Hub at runtime. |
| GPU server | TensorRT engine | FP16 inference, lowest latency |
| Edge / mobile | ONNX | ONNX Runtime available on ARM, iOS, Android |
| REST API | ONNX + FastAPI/Flask | Wrap the predict function in an HTTP endpoint |

### 9.3 CI/CD Flow

```
Push to main
    |
    +---> GitHub Actions: deploy-hf-spaces.yml
    |       |
    |       +---> Create HF Space (if needed)
    |       +---> Push app.py, Dockerfile, params.yaml, examples/
    |       +---> HF Spaces builds Docker container
    |
    +---> DVC Pipeline (manual): dvc repro
            |
            +---> push_to_hf stage uploads model.onnx to HF Hub
```

## 10. Pipeline Orchestration (DVC)

```
preprocessing → training → evaluate → log_production_model → export_onnx → push_to_hf
```

| Stage | Inputs | Outputs | Params |
|-------|--------|---------|--------|
| `preprocessing` | raw images, preprocess.py | processed images + .npy depth | data.img_size, baseline, focal_length, max_depth |
| `training` | processed data, model.py, data_loader.py, metrics.py | model.pth | data, model, training, augmentation, mlflow |
| `evaluate` | model.pth, test data, evaluate.py, metrics.py | metrics.json + MLflow test metrics | data, model, mlflow |
| `log_production_model` | model.pth, metrics.json, log_production_model.py | production_model.pth (conditional) | mlflow |
| `export_onnx` | production_model.pth, export_onnx.py, model.py | model.onnx | export, model |
| `push_to_hf` | model.onnx, push_to_hf.py | (uploads to HF Hub) | export.hf_repo_id |

DVC tracks:
- **Dependencies**: source code + data files that trigger re-runs on change
- **Parameters**: values from `params.yaml` that trigger re-runs on change
- **Outputs**: model.pth, production_model.pth, model.onnx (cached by DVC)
- **Metrics**: `metrics.json` (not cached, always readable)

## 11. Testing Strategy

```
tests/
├── test_model.py         # Architecture correctness
├── test_data_loader.py   # Data pipeline integrity
└── test_metrics.py       # Metric computation correctness
```

| Test | Purpose |
|------|---------|
| `test_forward_pass_shape` | Output shape matches input batch and spatial dims |
| `test_forward_pass_cpu` | Model runs on CPU without device errors |
| `test_output_non_negative` | ReLU constraint enforced |
| `test_different_decoder_channels` | Architecture generalizes to different channel configs |
| `test_dataset_length` | Dataset discovers all image files |
| `test_dataset_item_shapes` | Tensors have correct shape, dtype, and value range |
| `test_dataloader_batching` | DataLoader produces correct batch dimensions |
| `test_augmentation_shapes` | Augmented outputs maintain correct dimensions |
| `test_perfect_prediction` | Zero error when pred == gt |
| `test_metrics_keys` | All 7 standard metrics present |
| `test_empty_valid_mask` | Graceful handling when no valid pixels exist |

## 12. Configuration

All hyperparameters are centralized in `params.yaml`:

```yaml
data:          # paths, image size, batch size, num_workers, num_cities, max_depth, baseline, focal_length
model:         # decoder_features
training:      # epochs, learning rates, AMP, grad clip, scheduler, warmup, weight_decay, patience, loss weights
augmentation:  # flip prob, color jitter, crop scale
mlflow:        # server URI, experiment name, run name, model name
export:        # ONNX/TensorRT paths, opset version, hf_repo_id, hf_space_id
```

Single config file means:
- DVC detects parameter changes and re-runs affected stages
- MLflow logs the exact config used for each run
- No hardcoded values scattered across scripts
