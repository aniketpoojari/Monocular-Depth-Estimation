import glob
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import DepthDataset, get_data_loader

# Data is available only when both RGB images and .npy depth files exist
DATA_AVAILABLE = (
    os.path.exists("data/processed/depth_X/val")
    and len(glob.glob("data/processed/depth_y/val/**/*.npy", recursive=True)) > 0
)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Processed data not available")
def test_dataset_length():
    ds = DepthDataset("data/processed/depth_X/val", "data/processed/depth_y/val", img_size=224)
    assert len(ds) > 0, f"Expected non-empty dataset, got {len(ds)}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Processed data not available")
def test_dataset_item_shapes():
    ds = DepthDataset("data/processed/depth_X/val", "data/processed/depth_y/val", img_size=224)
    img, depth = ds[0]
    assert img.shape == (3, 224, 224), f"Image shape: {img.shape}"
    assert depth.shape == (1, 224, 224), f"Depth shape: {depth.shape}"
    assert img.dtype == torch.float32
    assert depth.min() >= 0
    assert depth.max() <= 80.0, f"Depth should be <= max_depth (80m), got max={depth.max()}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Processed data not available")
def test_dataloader_batching():
    loader = get_data_loader(
        "data/processed/depth_X/val", "data/processed/depth_y/val",
        img_size=224, batch_size=4, augment=False, num_workers=0,
    )
    images, depths = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)
    assert depths.shape == (4, 1, 224, 224)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Processed data not available")
def test_augmentation_shapes():
    aug_config = {
        "horizontal_flip_prob": 1.0,
        "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        "random_crop_scale": [0.8, 1.0],
    }
    ds = DepthDataset(
        "data/processed/depth_X/val", "data/processed/depth_y/val",
        img_size=224, augment=True, aug_config=aug_config,
    )
    img, depth = ds[0]
    assert img.shape == (3, 224, 224)
    assert depth.shape == (1, 224, 224)
