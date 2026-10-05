"""Feature maps ``f_map``: project node features ``x in R^d`` to ``h`` scalar fields.

Every map exposes a projection matrix ``W`` of shape ``(d, h)`` where this is
meaningful, which the sparsity regulariser and the mode-weight plots read.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NormalisedLinear(nn.Module):
    """Linear projection ``z = x W`` with unit-L2-norm columns.

    This is the default map of :class:`sail.SAILModes` when the Stiefel
    constraint is off, and the map every trained mode model is exported to
    (see :meth:`sail.SAILModes.fit`).

    Parameters
    ----------
    input_dim : int
        Input feature dimension ``d``.
    num_directions : int
        Number of projected fields ``h``.
    normalize : bool
        L2-normalise each column of ``W`` in the forward pass.
    trainable : bool
        If ``False`` the matrix is a fixed buffer.
    fixed_value : float or Tensor
        Initial value when ``trainable=False``; a float gives
        ``fixed_value * I[:d, :h]``.
    """

    def __init__(
        self,
        input_dim: int,
        num_directions: int = 1,
        normalize: bool = True,
        trainable: bool = True,
        fixed_value: float | Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.trainable = trainable
        if trainable:
            self._W = nn.Parameter(torch.randn(input_dim, num_directions))
        else:
            if not isinstance(fixed_value, Tensor):
                fixed_value = torch.eye(input_dim, num_directions) * float(fixed_value)
            self.register_buffer("_W", fixed_value.float())

    @property
    def W(self) -> Tensor:
        """Projection matrix, shape ``(d, h)`` (column-normalised if enabled)."""
        return F.normalize(self._W, p=2, dim=0) if self.normalize else self._W

    @W.setter
    def W(self, value: Tensor) -> None:
        with torch.no_grad():
            self._W.copy_(value)

    def forward(self, x: Tensor) -> Tensor:
        """``(N, d) -> (N, h)``."""
        return x @ self.W


class StiefelLinear(nn.Module):
    """Linear projection constrained to the Stiefel manifold (``W^T W = I``).

    Implemented with ``geotorch.orthogonal`` (matrix-exponential
    parametrisation). Used during training only: after
    :meth:`sail.SAILModes.fit` the learned matrix is exported to a
    :class:`NormalisedLinear`, whose forward pass is a plain matmul and is
    therefore much cheaper at inference time.

    Parameters
    ----------
    input_dim : int
        Input feature dimension ``d``.
    output_dim : int
        Number of orthonormal directions ``h <= d``.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        try:
            import geotorch
        except ImportError as err:  # pragma: no cover
            raise ImportError("StiefelLinear requires `geotorch` (pip install geotorch)") from err
        self._W = nn.Parameter(torch.randn(input_dim, output_dim))
        geotorch.orthogonal(self, "_W")

    @property
    def W(self) -> Tensor:
        """Orthonormal projection matrix, shape ``(d, h)``."""
        return self._W

    def forward(self, x: Tensor) -> Tensor:
        """``(N, d) -> (N, h)``."""
        return x @ self.W


class Linear(nn.Module):
    """Unconstrained affine projection ``z = x W + b`` (``nn.Linear``)."""

    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    @property
    def W(self) -> Tensor:
        """Projection matrix, shape ``(d, h)``."""
        return self.linear.weight.T

    def forward(self, x: Tensor) -> Tensor:
        """``(N, d) -> (N, h)``."""
        return self.linear(x)


class OneHotMap(nn.Module):
    """Concept-based projection: each node is assigned to one of ``m`` concepts.

    The forward pass uses the hard one-hot ``argmax`` encoding while gradients
    flow through the softmax (straight-through estimator), see Supplementary
    Section C-D.
    """

    def __init__(self, input_dim: int, num_concepts: int, straight_through: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_concepts)
        self.straight_through = straight_through

    @property
    def W(self) -> Tensor:
        return self.linear.weight.T

    def forward(self, x: Tensor) -> Tensor:
        """``(N, d) -> (N, m)`` one-hot (or soft, if ``straight_through=False``)."""
        soft = F.softmax(self.linear(x), dim=-1)
        if not self.straight_through:
            return soft
        hard = torch.zeros_like(soft).scatter_(1, soft.argmax(dim=-1, keepdim=True), 1.0)
        return hard + (soft - soft.detach())


class GumbelSoftmaxMap(nn.Module):
    """Gumbel-softmax relaxation of :class:`OneHotMap` (Supplementary Section C-C).

    Parameters
    ----------
    tau : float
        Gumbel temperature; as ``tau -> 0`` samples approach one-hot vectors.
    hard : bool
        Return straight-through one-hot samples.
    """

    def __init__(self, input_dim: int, num_concepts: int, tau: float = 1.0, hard: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_concepts)
        self.tau = tau
        self.hard = hard

    @property
    def W(self) -> Tensor:
        return self.linear.weight.T

    def forward(self, x: Tensor) -> Tensor:
        logits = self.linear(x)
        if not self.training:
            return F.softmax(logits / self.tau, dim=-1)
        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)


def orient_columns(W: Tensor) -> Tensor:
    """Flip the sign of each column so its largest-magnitude entry is positive.

    SAC objectives are invariant to the sign of a projection; fixing it makes
    learned modes comparable across runs.
    """
    idx = W.abs().argmax(dim=0)
    signs = torch.sign(W[idx, torch.arange(W.shape[1])])
    signs[signs == 0] = 1
    return W * signs


def get_projection(f_map: nn.Module) -> Optional[Tensor]:
    """Return ``f_map.W`` if the map has a projection matrix, else ``None``."""
    return getattr(f_map, "W", None)
