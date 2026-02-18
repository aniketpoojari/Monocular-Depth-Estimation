import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import compute_depth_metrics


def test_perfect_prediction():
    gt = torch.ones(1, 1, 10, 10) * 10.0  # 10 meters
    pred = gt.clone()
    m = compute_depth_metrics(pred, gt)
    assert m["abs_rel"] < 1e-5
    assert m["rmse"] < 1e-5
    assert m["delta_1"] > 0.99


def test_metrics_keys():
    pred = torch.rand(1, 1, 10, 10) * 50.0 + 1.0  # 1-51 meters
    gt = torch.rand(1, 1, 10, 10) * 50.0 + 1.0
    m = compute_depth_metrics(pred, gt)
    expected_keys = {"abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1", "delta_2", "delta_3"}
    assert set(m.keys()) == expected_keys


def test_empty_valid_mask():
    pred = torch.ones(1, 1, 5, 5) * 10.0
    gt = torch.zeros(1, 1, 5, 5)  # All below min_depth
    m = compute_depth_metrics(pred, gt)
    assert m["rmse"] == 0.0
