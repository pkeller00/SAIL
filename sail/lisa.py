"""LISA-style hotspot discovery on SAIL fields (Section III-D).

For each projected field ``z`` (a mode), the Moran-style local score
``z_j * lag_j`` with ``lag_j = sum_k w_jk z_k`` is tested against a
permutation null (``z`` shuffled across nodes, affinities held fixed).
Significant nodes are labelled by the signs of ``z_j`` and ``lag_j``.

The Moran-style score is used regardless of the local index the model was
trained with, because the High/Low quadrant labels need a signed statistic.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor

from .graph import SpatialGraph
from .model import SAIL

#: Integer id of each LISA category.
LISA_LABELS: dict[str, int] = {
    "Non-significant": 0,
    "High-High": 1,
    "Low-Low": 2,
    "Low-High": 3,
    "High-Low": 4,
}
LISA_NAMES: dict[int, str] = {v: k for k, v in LISA_LABELS.items()}

Correction = Union[Literal["BH", "maxT", "none"], Callable[[Tensor, Tensor, float], Tensor]]


def permutation_pvalues(
    observed: Tensor, permuted: Tensor, correction: Correction = "BH", alpha: float = 0.05
) -> Tensor:
    """Node-wise two-sided permutation p-values with multiple-testing correction.

    Parameters
    ----------
    observed : Tensor, shape (N,)
        Observed local statistic.
    permuted : Tensor, shape (P, N)
        Statistic under ``P`` permutations.
    correction : {"BH", "maxT", "none"} or callable
        ``"BH"``: Benjamini-Hochberg FDR on the raw p-values.
        ``"maxT"``: Westfall-Young max-T, comparing each node to the
        per-permutation maximum over nodes (controls FWER).
        A callable receives ``(|permuted|, |observed|, alpha)``.
    """
    obs, perm = observed.abs(), permuted.abs()
    if correction == "maxT":
        max_stats = perm.max(dim=1).values  # (P,)
        return (obs.unsqueeze(0) <= max_stats.unsqueeze(1)).float().mean(dim=0)
    if callable(correction):
        return correction(perm, obs, alpha)
    raw = (perm >= obs).float().mean(dim=0)
    if correction in ("BH", "fdr_bh"):
        from statsmodels.stats.multitest import multipletests

        return torch.as_tensor(multipletests(raw.numpy(), alpha=alpha, method="fdr_bh")[1])
    if correction == "none":
        return raw
    raise ValueError(f"Unknown correction {correction!r}")


def assign_labels(z: Tensor, lag: Tensor, pvalues: Tensor, alpha: float = 0.05) -> Tensor:
    """Quadrant labels (see :data:`LISA_LABELS`) for significant nodes, shape (N,)."""
    labels = torch.zeros_like(z, dtype=torch.long)
    sig = pvalues <= alpha
    labels[sig & (z > 0) & (lag > 0)] = LISA_LABELS["High-High"]
    labels[sig & (z < 0) & (lag < 0)] = LISA_LABELS["Low-Low"]
    labels[sig & (z < 0) & (lag > 0)] = LISA_LABELS["Low-High"]
    labels[sig & (z > 0) & (lag < 0)] = LISA_LABELS["High-Low"]
    return labels


@torch.no_grad()
def lisa(
    graph: SpatialGraph,
    model: SAIL,
    modes: Optional[Union[int, list[int]]] = None,
    permutations: int = 1000,
    alpha: float = 0.05,
    correction: Correction = "BH",
    generator: Optional[torch.Generator] = None,
) -> np.ndarray:
    """LISA cluster labels of every node for the requested modes.

    Parameters
    ----------
    graph : SpatialGraph
        A single graph.
    model : SAIL
        Trained model (``SAILModes.model`` or a supervised backbone).
    modes : int or list of int, optional
        Zero-based field indices; all fields if ``None``.
    permutations : int
        Size of the permutation null.
    alpha : float
        Significance level after correction.
    correction : {"BH", "maxT", "none"} or callable
        Multiple-testing correction across nodes.
    generator : torch.Generator, optional
        CPU RNG for the permutations; the global torch RNG if ``None``.

    Returns
    -------
    ndarray of int, shape (N,) if ``modes`` is an int, else (N, len(modes))
        Label ids, see :data:`LISA_LABELS`.
    """
    model.eval()
    out = model(graph)
    z_all = out.z[0].cpu()
    neighbors = out.neighbors[0].cpu()
    weights = out.weights[0].cpu()
    n, h = z_all.shape

    single = isinstance(modes, int)
    heads = [modes] if single else (list(range(h)) if modes is None else list(modes))
    labels = np.zeros((n, len(heads)), dtype=np.int64)

    for col, m in enumerate(heads):
        z = z_all[:, m]
        lag = (weights * z[neighbors]).sum(dim=1)
        observed = z * lag
        permuted = torch.empty(permutations, n)
        for p in range(permutations):
            z_perm = z[torch.randperm(n, generator=generator)]
            permuted[p] = z_perm * (weights * z_perm[neighbors]).sum(dim=1)
        pvals = permutation_pvalues(observed, permuted, correction, alpha)
        labels[:, col] = assign_labels(z, lag, pvals, alpha).numpy()

    return labels[:, 0] if single else labels
