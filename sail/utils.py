"""Small shared helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    """``cuda`` if available, else ``cpu``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
