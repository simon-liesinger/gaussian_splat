"""Loss functions for gaussian splat training."""

import torch
from pytorch_msssim import ssim as compute_ssim


def l1_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel L1 loss, averaged."""
    return (rendered - target).abs().mean()


def combined_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """Combined L1 + SSIM loss (as in original 3DGS paper).

    Args:
        rendered: [H, W, 3] or [1, 3, H, W] rendered image
        target: same shape as rendered
        lambda_ssim: weight for SSIM term (0.2 in original paper)
    """
    l1 = l1_loss(rendered, target)

    # pytorch_msssim expects [B, C, H, W]
    if rendered.ndim == 3:
        r = rendered.permute(2, 0, 1).unsqueeze(0)
        t = target.permute(2, 0, 1).unsqueeze(0)
    else:
        r = rendered
        t = target

    ssim_val = compute_ssim(r, t, data_range=1.0, size_average=True)

    return (1 - lambda_ssim) * l1 + lambda_ssim * (1 - ssim_val)
