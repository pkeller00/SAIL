"""Local spatial-autocorrelation operators ``f_local``.

Every operator has the signature ``(z, neighbors, weights) -> local`` with

* ``z``         : (N, h) standardised projected fields,
* ``neighbors`` : (N, K) neighbour indices,
* ``weights``   : (N, K) row-stochastic affinities (zero on padded slots),
* ``local``     : (N, h) local index per node and field.

With the identity map and fixed row-stochastic weights these reduce exactly to
the classical statistics (Theorem IV.1); see ``tests/test_classical_equivalence.py``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

LocalIndexFn = Callable[[Tensor, Tensor, Tensor], Tensor]


def spatial_lag(z: Tensor, neighbors: Tensor, weights: Tensor) -> Tensor:
    """Affinity-weighted neighbourhood mean ``sum_j w_ij z_j``, shape (N, h)."""
    return (weights.unsqueeze(-1) * z[neighbors]).sum(dim=1)


def moran(z: Tensor, neighbors: Tensor, weights: Tensor) -> Tensor:
    """Local Moran's I: ``z_i * sum_j w_ij z_j``."""
    return z * spatial_lag(z, neighbors, weights)


def geary(z: Tensor, neighbors: Tensor, weights: Tensor) -> Tensor:
    """Local Geary's C: ``1/2 * sum_j w_ij (z_i - z_j)^2``."""
    diff_sq = (z.unsqueeze(1) - z[neighbors]) ** 2
    return (weights.unsqueeze(-1) * diff_sq).sum(dim=1) / 2


def getis_ord_gstar(z: Tensor, neighbors: Tensor, weights: Tensor, eps: float = 1e-8) -> Tensor:
    """Standardised local Getis-Ord ``G*`` for standardised ``z``.

    ``G*_i = sum_j w_ij z_j / sqrt((N * sum_j w_ij^2 - (sum_j w_ij)^2) / (N - 1))``
    """
    n = z.size(0)
    row_sum = weights.sum(dim=1, keepdim=True)
    row_sq_sum = (weights**2).sum(dim=1, keepdim=True)
    denom = torch.sqrt((n * row_sq_sum - row_sum**2) / max(n - 1, 1) + eps)
    return spatial_lag(z, neighbors, weights) / denom


def morisita_horn(u: Tensor, neighbors: Tensor, weights: Tensor, eps: float = 1e-12) -> Tensor:
    """Morisita-Horn similarity between a node and its neighbourhood composition.

    Unlike the operators above this acts *jointly* on all channels: ``u`` must be
    a non-negative relative-abundance representation (rows sum to one; use an
    unstandardised map). Returns shape (N, 1).

    ``C_MH(u_i, v_i) = 2 <u_i, v_i> / (||u_i||^2 + ||v_i||^2)``
    with ``v_i = sum_j w_ij u_j``.
    """
    v = spatial_lag(u, neighbors, weights)
    num = 2 * (u * v).sum(dim=1, keepdim=True)
    den = (u**2).sum(dim=1, keepdim=True) + (v**2).sum(dim=1, keepdim=True)
    return num / (den + eps)


LOCAL_INDICES: dict[str, LocalIndexFn] = {
    "moran": moran,
    "geary": geary,
    "getis_ord_gstar": getis_ord_gstar,
    "gstar": getis_ord_gstar,
    "morisita_horn": morisita_horn,
}

#: Indices that act on each projected field independently and therefore
#: expect standardised inputs.
STANDARDISED_INDICES = {moran, geary, getis_ord_gstar}


def get_local_index(name_or_fn: str | LocalIndexFn) -> LocalIndexFn:
    """Resolve a local index by name (``"moran"``, ``"geary"``, ``"gstar"``, ...)."""
    if callable(name_or_fn):
        return name_or_fn
    key = name_or_fn.lower().removesuffix("_index")
    if key not in LOCAL_INDICES:
        raise ValueError(f"Unknown local index {name_or_fn!r}; choose from {sorted(LOCAL_INDICES)}")
    return LOCAL_INDICES[key]


def index_kind(fn: LocalIndexFn) -> str:
    """Family name of a local index (``"moran"``, ``"geary"``, ``"getis"``, ``"other"``)."""
    if fn is moran:
        return "moran"
    if fn is geary:
        return "geary"
    if fn is getis_ord_gstar:
        return "getis"
    return "other"
