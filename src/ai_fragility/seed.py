"""Global seeding utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed random, numpy, and torch for reproducibility.

    Args:
        seed: Desired seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


__all__: list[str] = ["set_global_seed"]
