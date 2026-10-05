"""Supervised SAIL (Section III-B, Supplementary Algorithm 1).

A SAIL backbone turns each graph into a vector of global SAC descriptors
``I_global`` (one per projected field, a "SAIL Head"); linear prediction heads
map the descriptors to the task output. The same heads applied to the
per-node projected fields give node-level read-outs for interpretation.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .feature_affinity import QueryKeyAffinity
from .feature_maps import Linear
from .graph import SpatialGraph
from .model import SAIL


class SAILPredictor(nn.Module):
    """SAIL backbone + ``n_outputs`` linear heads on the global descriptors.

    Parameters
    ----------
    in_dim : int
        Node feature dimension.
    n_heads : int
        Number of SAIL Heads (projected fields) ``h``.
    n_outputs : int
        Number of prediction heads (each ``Linear(h, 1)``).
    local_index : str
        Local SAC operator of the backbone.
    query_key_dim : int, optional
        Width of a query-key feature affinity added to the spatial logits;
        ``None`` for a purely spatial affinity.
    **sail_kwargs
        Further :class:`sail.SAIL` arguments (``reduce``, ``learnable_temperature``, ...).

    Notes
    -----
    Modules are created in the order affinity -> feature map -> heads, so
    seeded runs reproduce the pre-release implementation exactly.
    """

    def __init__(
        self,
        in_dim: int,
        n_heads: int = 16,
        n_outputs: int = 1,
        local_index: str = "geary",
        query_key_dim: Optional[int] = 16,
        **sail_kwargs,
    ) -> None:
        super().__init__()
        f_affinity = QueryKeyAffinity(in_dim, query_key_dim) if query_key_dim else None
        f_map = Linear(in_dim, n_heads)
        self.sail = SAIL(f_map=f_map, f_affinity=f_affinity, local_index=local_index, **sail_kwargs)
        self.heads = nn.ModuleList(nn.Linear(n_heads, 1) for _ in range(n_outputs))

    def descriptors(self, graph: SpatialGraph) -> tuple[Tensor, Tensor]:
        """Global descriptors ``(1, h)`` and standardised node fields ``(N, h)``."""
        out = self.sail(graph)
        return out.global_scores, out.z[0]

    def forward(self, graph: SpatialGraph) -> tuple[Tensor, Tensor]:
        """Graph-level output ``(1, n_outputs)`` and node-level read-out ``(N, n_outputs)``."""
        g, z = self.descriptors(graph)
        return (torch.cat([h(g) for h in self.heads], dim=1),
                torch.cat([h(z) for h in self.heads], dim=1))

    def head_weights(self) -> Tensor:
        """Weights of the prediction heads, shape ``(n_outputs, h)``."""
        return torch.cat([h.weight for h in self.heads], dim=0).detach()


def load_state_dict_compat(model: SAILPredictor, state: dict, head_names: Union[list[str], None] = None) -> None:
    """Load a pre-release dual-head state dict (``sail.f_theta.*``, ``headE``/``headS``).

    ``head_names`` lists the old head attribute names in output order.
    """
    head_names = head_names or ["headE", "headS"]
    new = {}
    for k, v in state.items():
        k = k.replace("sail.f_theta.", "sail.f_affinity.")
        for i, name in enumerate(head_names):
            if k.startswith(name + "."):
                k = f"heads.{i}." + k[len(name) + 1:]
        new[k] = v
    model.load_state_dict(new)
