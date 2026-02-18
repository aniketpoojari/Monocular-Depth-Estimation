import torch


def compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=80.0):
    """Compute standard monocular depth estimation metrics.

    Args:
        pred: predicted depth (B, 1, H, W)
        gt: ground truth depth (B, 1, H, W)
        min_depth: minimum valid depth
        max_depth: maximum valid depth in meters

    Returns:
        dict with abs_rel, sq_rel, rmse, rmse_log, delta_1, delta_2, delta_3
    """
    valid_mask = gt > min_depth

    pred = pred[valid_mask]
    gt = gt[valid_mask]

    if pred.numel() == 0:
        return {
            "abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0,
            "rmse_log": 0.0, "delta_1": 0.0, "delta_2": 0.0, "delta_3": 0.0,
        }

    pred = torch.clamp(pred, min=min_depth, max=max_depth)

    thresh = torch.max(pred / gt, gt / pred)
    delta_1 = (thresh < 1.25).float().mean().item()
    delta_2 = (thresh < 1.25**2).float().mean().item()
    delta_3 = (thresh < 1.25**3).float().mean().item()

    abs_rel = (torch.abs(pred - gt) / gt).mean().item()
    sq_rel = (((pred - gt) ** 2) / gt).mean().item()
    rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
    rmse_log = torch.sqrt(
        ((torch.log(pred + 1e-8) - torch.log(gt + 1e-8)) ** 2).mean()
    ).item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }
