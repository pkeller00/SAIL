"""The SAIL operator (Algorithm 1 of the paper).

For a graph with node features ``X`` the forward pass computes::

    z       = standardise(f_map(X))                       (N, h)
    w_ij    = softmax_j(-tau * d_ij - f_affinity(x_i, x_j)) (N, K)
    I_local = f_act(f_local(z, neighbours, w))             (N, h)
    I_global= f_reduce(I_local over nodes)                 (h,)

All graph types (kNN, radius, dense) share this single forward pass via the
padded-neighbour layout of :class:`sail.SpatialGraph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph import SpatialGraph
from .local_indices import STANDARDISED_INDICES, LocalIndexFn, get_local_index

Standardise = Literal["population", "sample", "none"]
ReduceFn = Callable[..., Tensor]


@dataclass
class SAILOutput:
    """Result of :meth:`SAIL.forward` for a batch of ``B`` graphs.

    Attributes
    ----------
    global_scores : Tensor, shape (B, h)
        Global SAIL score of every projected field, per graph.
    local : list of Tensor, each (N_b, h)
        Local index per node.
    z : list of Tensor, each (N_b, h)
        Standardised projected fields ``f_map(x)``.
    weights : list of Tensor, each (N_b, K_b)
        Row-stochastic affinities (zero on padded slots and self-loops).
    neighbors : list of Tensor, each (N_b, K_b)
        Neighbour indices the weights refer to.
    """

    global_scores: Tensor
    local: list[Tensor] = field(default_factory=list)
    z: list[Tensor] = field(default_factory=list)
    weights: list[Tensor] = field(default_factory=list)
    neighbors: list[Tensor] = field(default_factory=list)


def reduce_mean(x: Tensor, dim: int = 0) -> Tensor:
    return x.mean(dim=dim)


def reduce_sum(x: Tensor, dim: int = 0) -> Tensor:
    return x.sum(dim=dim)


def reduce_max(x: Tensor, dim: int = 0) -> Tensor:
    return x.max(dim=dim).values


REDUCTIONS: dict[str, ReduceFn] = {"mean": reduce_mean, "sum": reduce_sum, "max": reduce_max}


class SAIL(nn.Module):
    """Spatial Autocorrelation Index Layer.

    Parameters
    ----------
    f_map : nn.Module, optional
        Feature map ``(N, d) -> (N, h)``. Identity if ``None``.
    f_affinity : nn.Module, optional
        Learned feature affinity ``((N, K, d), (N, K, d)) -> (N, K)``
        subtracted from the attention logits (see :mod:`sail.feature_affinity`).
        ``None`` (paper default) gives a purely distance-based affinity.
    local_index : str or callable
        Local operator ``f_local``: ``"geary"``, ``"moran"``, ``"gstar"``,
        ``"morisita_horn"`` or any callable ``(z, neighbors, weights) -> (N, h)``.
    reduce : {"mean", "sum", "max"} or callable
        Global reduction ``f_reduce`` over nodes.
    f_act : nn.Module
        Activation applied to the local index before reduction.
    init_temp : float
        Initial value of the raw temperature; the effective spatial
        temperature is ``softplus(raw)``.
    learnable_temperature : bool
        Learn ``tau`` (default) or keep it fixed.
    standardise : {"population", "sample", "none"}
        Column standardisation of ``f_map(x)``. ``"population"`` (divide by the
        biased std) is the paper default and recovers Moran's I exactly;
        ``"sample"`` recovers Geary's C; ``"none"`` is required for
        Morisita-Horn.
    self_loops : bool
        Allow ``w_ii > 0`` when a node appears in its own neighbour list.
    mode_chunk : int, optional
        Compute the local index for at most this many projected fields at a
        time. Results are identical; peak memory of the ``(N, K, h)`` neighbour
        gather drops by ``h / mode_chunk``. Only for per-field indices
        (Moran, Geary, G*). Can be changed after construction.
    """

    def __init__(
        self,
        f_map: Optional[nn.Module] = None,
        f_affinity: Optional[nn.Module] = None,
        local_index: Union[str, LocalIndexFn] = "moran",
        reduce: Union[str, ReduceFn] = "mean",
        f_act: Optional[nn.Module] = None,
        init_temp: float = 1.0,
        learnable_temperature: bool = True,
        standardise: Standardise = "population",
        self_loops: bool = False,
        mode_chunk: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.f_map = f_map if f_map is not None else nn.Identity()
        self.f_affinity = f_affinity
        self.local_index = get_local_index(local_index)
        self.reduce = REDUCTIONS[reduce] if isinstance(reduce, str) else reduce
        self.f_act = f_act if f_act is not None else nn.Identity()
        self.standardise = standardise
        self.self_loops = self_loops
        self.mode_chunk = mode_chunk

        raw = torch.tensor(float(init_temp), dtype=torch.float32)
        if learnable_temperature:
            self.raw_temperature = nn.Parameter(raw)
        else:
            self.register_buffer("raw_temperature", raw)

    # ------------------------------------------------------------------ #
    @property
    def temperature(self) -> Tensor:
        """Effective spatial temperature ``tau = softplus(raw_temperature)``."""
        return F.softplus(self.raw_temperature)

    @property
    def device(self) -> torch.device:
        return self.raw_temperature.device

    def project(self, x: Tensor) -> Tensor:
        """Apply ``f_map`` and the configured standardisation: ``(N, d) -> (N, h)``."""
        z = self.f_map(x)
        if self.standardise == "none":
            return z
        z = z - z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True, unbiased=self.standardise == "sample")
        return z / (std + 1e-8)

    def affinity(
        self, x: Tensor, neighbors: Tensor, distances: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Row-stochastic affinities over the padded neighbourhood, shape (N, K).

        ``w_ij = softmax_j(-tau * d_ij - f_affinity(x_i, x_j))``, with padded
        slots (``mask == False``) and, unless ``self_loops``, self-neighbours
        set to zero weight.
        """
        logits = -self.temperature * distances
        if self.f_affinity is not None:
            x_j = x[neighbors]
            x_i = x.unsqueeze(1).expand_as(x_j)
            logits = logits - self.f_affinity(x_i, x_j)

        invalid = None if mask is None else ~mask
        if not self.self_loops:
            is_self = neighbors == torch.arange(neighbors.size(0), device=neighbors.device).unsqueeze(1)
            invalid = is_self if invalid is None else invalid | is_self
        if invalid is not None:
            logits = logits.masked_fill(invalid, float("-inf"))

        weights = torch.softmax(logits, dim=1)
        if invalid is not None:
            # nodes without any valid neighbour get all-zero weights, not NaN
            weights = torch.nan_to_num(weights, nan=0.0)
        return weights

    def local_scores(self, z: Tensor, neighbors: Tensor, weights: Tensor) -> Tensor:
        """Local index ``(N, h)``, optionally evaluated ``mode_chunk`` fields at a time."""
        c = self.mode_chunk
        if c is None or c >= z.shape[1] or self.local_index not in STANDARDISED_INDICES:
            return self.local_index(z, neighbors, weights)
        return torch.cat([self.local_index(z[:, i : i + c], neighbors, weights)
                          for i in range(0, z.shape[1], c)], dim=1)

    # ------------------------------------------------------------------ #
    def forward(self, graphs: Union[SpatialGraph, Sequence[SpatialGraph]]) -> SAILOutput:
        """Run SAIL on one graph or a list of graphs.

        Graphs are processed independently (each has its own standardisation
        and neighbourhood) and their global scores are stacked.

        Any object with ``x``, ``neighbors`` and ``distances`` attributes (and
        optionally ``mask``) is accepted, e.g. a ``torch_geometric`` ``Data``.
        """
        graphs = [graphs] if not isinstance(graphs, (list, tuple)) else graphs
        device = self.device
        out = SAILOutput(global_scores=torch.empty(0))
        scores = []
        for g in graphs:
            x = g.x.to(device)
            neighbors = g.neighbors.to(device).long()
            distances = g.distances.to(device)
            mask = getattr(g, "mask", None)
            mask = None if mask is None else mask.to(device)

            z = self.project(x)
            weights = self.affinity(x, neighbors, distances, mask)
            local = self.f_act(self.local_scores(z, neighbors, weights))
            scores.append(self.reduce(local, dim=0))

            out.local.append(local)
            out.z.append(z)
            out.weights.append(weights)
            out.neighbors.append(neighbors)
        out.global_scores = torch.stack(scores, dim=0)
        return out
