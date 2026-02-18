# System Design: Monocular Depth Estimation

## 1. Overview

This system estimates per-pixel depth from a single RGB image using a Dense Prediction Transformer (DPT). It covers the full ML lifecycle: data preprocessing, model training, experiment tracking, evaluation, model export, and serving via a web demo.

## 2. System Architecture

```
+---------------------+       +---------------------+       +---------------------+
|    Data Ingestion    |       |   Training Pipeline |       |     Serving Layer   |
|                     |       |                     |       |                     |
| Cityscapes Dataset  +------>+ DVC Pipeline        +------>+ Gradio App (ONNX)   |
| (leftImg8bit +      |       | (preprocess, train, |       | or                  |
|  disparity maps)    |       |  evaluate, log)     |       | TensorRT Engine     |
+---------------------+       +----------+----------+       +---------------------+
                                         |
                                         v
                               +-------------------+
                               |  MLflow Tracking   |
                               |  (SQLite backend)  |
                               +-------------------+
```

## 3. Data Flow

### 3.1 Raw Data

```
data/raw/
├── depth_X/{train,val,test}/{city}/*.png    # RGB images (variable resolution)
└── depth_y/{train,val,test}/{city}/*.png    # Disparity maps (uint16)
```

Source: [Cityscapes dataset](https://www.cityscapes-dataset.com/) with leftImg8bit (RGB) and disparity packages.

### 3.2 Preprocessing

**Script**: `src/preprocess.py`

```
Raw images (1024x2048) --> Resize to 224x224 --> Processed images
```

- RGB images: resized with BICUBIC interpolation
- Depth maps: resized with NEAREST interpolation (preserves discrete disparity values)
- Skips already-processed cities for incremental runs
- Controlled by `num_cities` param for subset training

### 3.3 Data Loading

**Script**: `src/data_loader.py`

```
Processed PNG --> PIL.Image --> Augmentation --> Tensor --> Normalize --> Batch
```

- **Image normalization**: ImageNet mean/std (required for pretrained ViT)
- **Depth conversion**: Cityscapes disparity (uint16) → metric depth in meters via `depth = baseline × focal_length / disparity`, clamped to [0, max_depth]. Camera intrinsics (baseline, focal_length) and max_depth are configurable in `params.yaml`.
- **Why metric depth instead of normalized disparity?** The previous approach (`disparity / 22500 → [0, 1]`) had several problems: (1) 22500 was a hardcoded magic number specific to Cityscapes, (2) disparity is inversely proportional to depth so linear normalization distorts the depth distribution, (3) Sigmoid output saturates at extremes causing vanishing gradients for very close/far objects. Metric depth in meters is physically meaningful, dataset-agnostic, and works naturally with ReLU output.
- **Augmentations** (train only): horizontal flip, color jitter, random resized crop
- Augmentations are applied consistently to image-depth pairs (same flip, same crop)
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
              (skip connections + upsample)
                           |
                    Output Head
              (Conv -> Upsample -> Conv -> ReLU)
                           |
                  Depth Map (B, 1, 224, 224)
```

### 4.2 Components

| Component | Description | Parameters |
|-----------|-------------|------------|
| **ViT-Base/16** | Pretrained encoder from timm. Patch size 16, embed dim 768, 12 transformer blocks | ~86M |
| **ReassembleLayer** (x4) | Projects 768-dim tokens to decoder channels, resamples to target spatial resolution via bilinear interpolation + Conv2d | ~3M |
| **FusionBlock** (x4) | Adds skip connection, applies Conv + BatchNorm + ReLU, then upsamples 2x via bilinear interpolation + Conv2d | ~1M |
| **Output Head** | 3-layer conv head: channels -> channels/2 -> 32 -> 1, with bilinear upsample and ReLU | ~0.1M |

**Total**: ~90M parameters, ~357 MB (ONNX)

### 4.3 Design Decisions

- **Why ViT over CNN encoder?** ViT captures global context in every layer via self-attention, which is critical for depth estimation where distant objects inform relative scale.
- **Why extract at layers 3, 6, 9, 12?** Provides multi-scale representations — early layers capture local texture, later layers capture global structure.
- **Why ReLU on output?** The model now predicts metric depth in meters (0 to 80m), which is unbounded above zero. ReLU enforces non-negativity without the gradient saturation problem of Sigmoid. Previously, Sigmoid was used when depth was normalized to [0, 1], but this caused vanishing gradients at depth extremes and required a hardcoded normalization constant.
- **Why scale-invariant + gradient loss instead of RMSE?** The previous loss (`sqrt(MSE)`) treated all depth errors equally regardless of distance. The new combined loss addresses this: (1) **Scale-invariant log loss** (Eigen et al., 2014) focuses on relative depth relationships — a 1m error at 5m depth is penalized more than at 50m. Formula: `mean(d²) - λ·mean(d)²` where `d = log(pred) - log(gt)`. (2) **Gradient loss** penalizes differences in spatial gradients between predicted and ground truth depth, producing sharper edges at object boundaries. The total loss is `si_weight × SI_loss + grad_weight × grad_loss`, with weights configurable in `params.yaml`.
- **Why bilinear upsample + Conv2d instead of ConvTranspose2d?** `ConvTranspose2d` produces checkerboard/box artifacts when the kernel size is not evenly divisible by the stride (e.g., kernel=3, stride=4). This is a well-known issue in dense prediction models — the uneven overlap of transposed convolution kernels creates a grid pattern in the output that aligns with ViT patch boundaries (16x16), making it especially visible. The fix is `nn.Upsample(bilinear) + nn.Conv2d(3x3)`: bilinear interpolation produces smooth upsampling, then the 3x3 convolution learns to refine spatial details without introducing artifacts. This pattern is used by the original DPT paper and most modern dense prediction architectures.
- **Why BatchNorm + ReLU in FusionBlock?** The original FusionBlock was a bare add + upsample with no normalization. Adding BatchNorm stabilizes decoder training across varying feature magnitudes, and ReLU introduces non-linearity that helps blend skip connection features before upsampling, producing smoother depth transitions.

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
| Optimizer | AdamW | Weight decay regularization prevents overfitting, standard for transformers |
| Weight decay | 0.01 | Regularizes decoder weights; ViT encoder benefits from decoupled weight decay |
| ViT LR | 1e-5 | Low LR for pretrained encoder to avoid catastrophic forgetting |
| Decoder LR | 1e-4 | Higher LR for randomly initialized decoder |
| Scheduler | Cosine annealing + linear warmup | Warmup stabilizes early training; cosine decay avoids sudden LR drops |
| Warmup epochs | 3 | Gradual LR increase from 0.1x to 1x |
| AMP | Enabled | FP16 forward pass reduces memory and increases throughput on CUDA |
| Gradient clipping | 1.0 | Prevents exploding gradients from transformer attention |
| Loss | Scale-invariant log + gradient | SI loss captures relative depth; gradient loss sharpens edges |
| Early stopping | patience=10 | Stops training if val RMSE doesn't improve for 10 consecutive epochs |

**Why AdamW instead of Adam?** Adam applies weight decay as L2 regularization coupled with the adaptive learning rate, which is suboptimal — large gradients reduce the effective regularization. AdamW decouples weight decay from the gradient update, providing consistent regularization regardless of gradient magnitude. This is the standard optimizer for transformer fine-tuning (used by BERT, GPT, DPT).

**Why early stopping?** Without it, the model may overfit after the learning rate decays to near zero. Early stopping saves compute and prevents the model from memorizing training noise.

### 5.3 Differential Learning Rates

The encoder (ViT) and decoder use separate learning rates:

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
| RMSE | sqrt(mean((pred - gt)^2)) | Absolute accuracy |
| RMSE log | sqrt(mean((log(pred) - log(gt))^2)) | Scale-invariant accuracy |
| delta < 1.25 | % of pixels where max(pred/gt, gt/pred) < 1.25 | Inlier ratio (strict) |
| delta < 1.25^2 | Same with threshold 1.5625 | Inlier ratio (moderate) |
| delta < 1.25^3 | Same with threshold 1.953125 | Inlier ratio (relaxed) |

These are the standard metrics from [Eigen et al., 2014](https://arxiv.org/abs/1406.2283), used across all monocular depth estimation benchmarks.

## 7. Experiment Tracking

**Tool**: MLflow with SQLite backend

### 7.1 What Gets Tracked

- **Parameters**: batch size, learning rates, epochs, image size, AMP, grad clip norm, weight decay, patience, loss weights, max depth
- **Metrics per epoch**: train loss, val RMSE, val AbsRel, val delta_1, learning rate
- **Artifacts**: Full PyTorch model (via `mlflow.pytorch.log_model`)
- **Model Registry**: Best model registered as `ViT+Decoder` with versioning

### 7.2 Production Model Selection

**Script**: `src/log_production_model.py`

Queries MLflow for all completed runs, selects the run with the lowest `val_rmse`, and downloads the model artifact to `saved_models/model.pth`.

## 8. Model Export

### 8.1 ONNX Export

**Script**: `src/export_onnx.py`

```
PyTorch model --> torch.onnx.export --> ONNX model --> Verification
```

- Opset version 14
- Dynamic batch axis
- Verified by comparing PyTorch and ONNX Runtime outputs (threshold: max diff < 1e-3)
- Output: `saved_models/model.onnx` (~357 MB)

### 8.2 TensorRT Export (Optional)

**Script**: `src/export_tensorrt.py`

```
ONNX model --> TensorRT builder --> Serialized engine
```

- FP16 mode for GPU-optimized inference
- 1 GB workspace limit
- Requires NVIDIA TensorRT SDK and pycuda

## 9. Serving

### 9.1 Gradio Web App

**Script**: `app.py`

```
User uploads image
        |
   Preprocess (resize 224x224, normalize)
        |
   ONNX Runtime inference
        |
   Colorize depth map (inferno colormap)
        |
   Display predicted depth
```

- Uses ONNX Runtime (CPU-compatible, no GPU required for demo)
- Lazy-loads the ONNX session on first request
- Includes example images for quick testing

### 9.2 Deployment Options

| Target | Format | Notes |
|--------|--------|-------|
| Hugging Face Spaces | ONNX + Gradio | Push app.py, model.onnx, requirements.txt, examples/ |
| GPU server | TensorRT engine | FP16 inference, lowest latency |
| Edge / mobile | ONNX | ONNX Runtime available on ARM, iOS, Android |
| REST API | ONNX + FastAPI/Flask | Wrap the predict function in an HTTP endpoint |

## 10. Pipeline Orchestration (DVC)

```yaml
preprocessing --> training --> evaluate --> log_production_model
```

| Stage | Inputs | Outputs | Params |
|-------|--------|---------|--------|
| `preprocessing` | raw images, preprocess.py | processed images | data.img_size |
| `training` | processed images, model.py, data_loader.py, metrics.py | model.pth | data, model, training, augmentation, mlflow |
| `evaluate` | model.pth, test data, evaluate.py, metrics.py | metrics.json | data, model |
| `log_production_model` | model.pth, log_production_model.py | (side effect) | mlflow |

DVC tracks:
- **Dependencies**: source code + data files that trigger re-runs on change
- **Parameters**: values from `params.yaml` that trigger re-runs on change
- **Outputs**: `saved_models/model.pth` (cached by DVC)
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
export:        # ONNX/TensorRT paths, opset version
```

Single config file means:
- DVC detects parameter changes and re-runs affected stages
- MLflow logs the exact config used for each run
- No hardcoded values scattered across scripts
