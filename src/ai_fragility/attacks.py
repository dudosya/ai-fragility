"""Adversarial attack implementations."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


ForwardFn = Callable[[torch.Tensor], torch.Tensor]


def fgsm_attack(
    forward_fn: ForwardFn,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Fast Gradient Sign Method (untargeted) attack.

    Args:
        model: Model to attack.
        images: Normalized input batch (NCHW) requiring gradients.
        labels: Ground-truth or proxy labels for loss calculation.
        epsilon: Perturbation magnitude (L-infinity ball).

    Returns:
        Adversarially perturbed images clipped to [0, 1] in data space.
    """

    images = images.clone().detach().requires_grad_(True)
    logits = forward_fn(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    adv_images = torch.clamp(images + perturbation, 0.0, 1.0)
    return adv_images.detach()


def pgd_attack(
    forward_fn: ForwardFn,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    step_callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Projected Gradient Descent (L-infinity) attack.

    Args:
        model: Model to attack.
        images: Normalized input batch (NCHW).
        labels: Ground-truth or proxy labels.
        epsilon: Maximum perturbation magnitude.
        alpha: Step size for each PGD iteration.
        steps: Number of optimization steps.
        step_callback: Optional hook executed each step with (step_idx, adv_images).

    Returns:
        Adversarially perturbed images clipped to [0, 1].
    """

    adv_images = images.clone().detach()
    adv_images = torch.clamp(adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon), 0.0, 1.0)

    for step in range(steps):
        adv_images.requires_grad_(True)
        logits = forward_fn(adv_images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            grad_sign = adv_images.grad.sign()
            adv_images = adv_images + alpha * grad_sign
            adv_images = torch.max(torch.min(adv_images, images + epsilon), images - epsilon)
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

        if step_callback is not None:
            step_callback(step, adv_images)

    return adv_images.detach()


__all__: list[str] = ["ForwardFn", "fgsm_attack", "pgd_attack"]
