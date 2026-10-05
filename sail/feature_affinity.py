"""Feature affinities ``f_affinity(x_i, x_j)``.

A feature affinity is an optional, learned pairwise term that is *subtracted*
from the spatial attention logits, so that the affinity between nodes ``i``
and ``j`` depends on their features as well as on their distance::

    logit_ij = -tau * d_ij - f_affinity(x_i, x_j)

Inputs are gathered neighbour tensors of shape ``(N, K, d)``; the output is
``(N, K)``.
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch import Tensor


class SharedMapAffinity(nn.Module):
    """Squared distance in the projected space: ``||f_map(x_i) - f_map(x_j)||^2``.

    Re-uses the model's own feature map, so it adds no parameters.
    """

    def __init__(self, f_map: nn.Module) -> None:
        super().__init__()
        self.f_map = f_map

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        """``(N, K, d), (N, K, d) -> (N, K)``."""
        return ((self.f_map(x_i) - self.f_map(x_j)) ** 2).sum(dim=-1)


class QueryKeyAffinity(nn.Module):
    """Scaled dot-product affinity ``q(x_i)^T k(x_j) / sqrt(r)``.

    Parameters
    ----------
    in_dim : int
        Input feature dimension ``d``.
    hidden_dim : int
        Width ``r`` of the query and key projections (paper: 16).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        """``(N, K, d), (N, K, d) -> (N, K)``."""
        return (self.query(x_i) * self.key(x_j)).sum(dim=-1) / math.sqrt(self.hidden_dim)
