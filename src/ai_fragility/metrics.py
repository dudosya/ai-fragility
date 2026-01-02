"""Metrics for adversarial robustness demos."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SecurityGap:
    """Represents the delta between attacked and defended confidence."""

    clean_confidence: float
    attacked_confidence: float
    defended_confidence: float
    delta: float


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """Compute probabilities from logits."""

    return F.softmax(logits, dim=-1)


def compute_security_gap(
    clean_logits: torch.Tensor,
    attacked_logits: torch.Tensor,
    defended_logits: torch.Tensor,
    reference_class: int,
) -> SecurityGap:
    """Compute confidence changes for a reference class across stages.

    Args:
        clean_logits: Model outputs on the clean input (1 x C).
        attacked_logits: Outputs on attacked input (1 x C).
        defended_logits: Outputs after defense (1 x C).
        reference_class: Class index to track (usually the clean top-1).

    Returns:
        Structured security gap metrics.
    """

    clean_prob = softmax_probs(clean_logits)[0, reference_class].item()
    attacked_prob = softmax_probs(attacked_logits)[0, reference_class].item()
    defended_prob = softmax_probs(defended_logits)[0, reference_class].item()
    delta = defended_prob - attacked_prob

    return SecurityGap(
        clean_confidence=clean_prob,
        attacked_confidence=attacked_prob,
        defended_confidence=defended_prob,
        delta=delta,
    )


__all__: list[str] = ["SecurityGap", "softmax_probs", "compute_security_gap"]
