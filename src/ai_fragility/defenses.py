"""Input sanitation defenses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

DefenseKind = Literal["gaussian", "median", "jpeg", "identity"]


@dataclass
class DefenseConfig:
    """Configuration for the assurance layer."""

    kind: DefenseKind = "gaussian"
    kernel_size: int = 3
    sigma: float = 1.0
    jpeg_quality: int = 75


def _ensure_pil(image: torch.Tensor) -> Image.Image:
    image_cpu = image.detach().cpu()
    return TF.to_pil_image(image_cpu)


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = TF.to_tensor(image).to(device)
    return tensor.clamp(0.0, 1.0)


def apply_defense(images: torch.Tensor, config: DefenseConfig, device: torch.device) -> torch.Tensor:
    """Apply a deterministic defense to a batch of images.

    Args:
        images: Input images in [0, 1].
        config: Defense configuration.
        device: Target device for the returned tensor.

    Returns:
        Defended images tensor.
    """

    if config.kind == "identity":
        return images.clone().detach()

    defended: list[torch.Tensor] = []
    for image in images:
        pil_image = _ensure_pil(image)
        if config.kind == "gaussian":
            filtered = pil_image.filter(ImageFilter.GaussianBlur(radius=config.sigma))
        elif config.kind == "median":
            filtered = pil_image.filter(ImageFilter.MedianFilter(size=config.kernel_size))
        elif config.kind == "jpeg":
            filtered = _jpeg_filter(pil_image, quality=config.jpeg_quality)
        else:
            msg = f"Unsupported defense kind {config.kind}"
            raise ValueError(msg)

        defended.append(_pil_to_tensor(filtered, device))

    return torch.stack(defended, dim=0)


def _jpeg_filter(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG round-trip to attenuate high-frequency adversarial noise."""

    from io import BytesIO

    with BytesIO() as handle:
        image.convert("RGB").save(handle, format="JPEG", quality=quality)
        handle.seek(0)
        buffered = Image.open(handle).convert("RGB").copy()
    return buffered


__all__: list[str] = ["DefenseKind", "DefenseConfig", "apply_defense"]
