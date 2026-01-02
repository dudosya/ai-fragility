"""Visualization helpers."""

from __future__ import annotations

import torch
from PIL import Image
import numpy as np


def gradient_to_heatmap(gradient: torch.Tensor) -> Image.Image:
    """Convert an input gradient to a heatmap image.

    Args:
        gradient: Input gradient tensor shaped (C, H, W).

    Returns:
        PIL image heatmap scaled to 0-255.
    """

    grad_cpu = gradient.detach().cpu()
    grad_abs = grad_cpu.abs()
    grad_max = grad_abs.max()
    if grad_max.item() == 0:
        grad_norm = grad_abs
    else:
        grad_norm = grad_abs / grad_max

    heatmap = torch.stack([grad_norm.mean(dim=0)] * 3, dim=0)
    heatmap_np = (heatmap.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.transpose(heatmap_np, (1, 2, 0)))


__all__: list[str] = ["gradient_to_heatmap"]
