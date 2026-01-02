"""Model loading utilities for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class ModelBundle:
    """Container for model, preprocessing, and metadata."""

    model: torch.nn.Module
    to_tensor: Callable[[object], torch.Tensor]
    normalize: Callable[[torch.Tensor], torch.Tensor]
    categories: list[str]

    def preprocess(self, image: object) -> torch.Tensor:  # type: ignore[override]
        """Convert an input image into a normalized tensor for the model."""

        tensor = self.to_tensor(image)
        return self.normalize(tensor)


def load_model(model_name: str, device: torch.device) -> ModelBundle:
    """Load a pretrained vision model and its preprocessing.

    Args:
        model_name: Name of the model to load. Currently supports "resnet18".
        device: Torch device for the model.

    Returns:
        Model bundle with the model on the specified device and its preprocessing transform.
    """

    if model_name != "resnet18":
        msg = f"Unsupported model {model_name}. Try 'resnet18'."
        raise ValueError(msg)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device)
    model.eval()

    weight_transforms = weights.transforms()

    to_tensor = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    normalize = transforms.Normalize(mean=weight_transforms.mean, std=weight_transforms.std)

    categories = list(getattr(weights, "categories", []) or weights.meta.get("categories", []))

    return ModelBundle(
        model=model,
        to_tensor=to_tensor,
        normalize=normalize,
        categories=categories,
    )


__all__: list[str] = ["ModelBundle", "load_model"]
