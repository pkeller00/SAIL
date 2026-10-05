"""Moran's I regression benchmark (Section V-A, Fig. 3; Supplementary Section G).

Each sample is a random point cloud with Gaussian-process node features. The
target is Moran's I of a *hidden* random projection of the features on a
*hidden* kNN neighbourhood (RBF edge weights), so it cannot be memorised and
requires modelling spatial autocorrelation. No model sees the generating
neighbourhood: SAIL receives all pairwise distances (dense graph, learned
temperature); GCN/GAT receive the fully connected, distance-weighted graph.

The data generator consumes Python's and torch's global RNGs in the same order
as the original implementation, so a given seed yields identical datasets.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sail
from sail.supervised import SAILPredictor


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class RegressionGraph:
    """One sample: SAIL's dense graph plus the inputs of the message-passing baselines."""

    graph: sail.SpatialGraph            # dense (all pairs), raw distances
    coords: torch.Tensor                # (N, 2)
    edge_index_fc: torch.Tensor         # (2, N(N-1)) fully connected, no self loops
    edge_weight_fc: torch.Tensor        # RBF weights of the edges
    edge_attr_fc: torch.Tensor          # raw distances of the edges, (E, 1)
    y_spatial: float                    # Moran's I of the hidden projection
    y_feat: float = 0.0                 # content target (blend ablation)
    y: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    x_baseline: Optional[torch.Tensor] = None

    @property
    def x(self) -> torch.Tensor:
        return self.graph.x


def rbf(dist: torch.Tensor, lengthscale: float) -> torch.Tensor:
    return torch.exp(-(dist**2) / (2.0 * lengthscale**2))


def knn_edges(coords: torch.Tensor, k: int):
    """Symmetrised kNN edge list (self excluded) and edge distances."""
    n = coords.size(0)
    k = max(1, min(k, n - 1))
    D = torch.cdist(coords, coords)
    vals, idx = torch.topk(-D, k=k, dim=1)
    row = torch.arange(n).unsqueeze(1).repeat(1, k).flatten()
    col, dist = idx.flatten(), (-vals).flatten()
    keep = row != col
    row, col, dist = row[keep], col[keep], dist[keep]
    return torch.stack([torch.cat([row, col]), torch.cat([col, row])]), torch.cat([dist, dist])


@torch.no_grad()
def morans_i(coords: torch.Tensor, z: torch.Tensor, k: int, lengthscale: float) -> float:
    """Classical Moran's I of ``z`` on the symmetrised kNN graph with RBF weights."""
    (row, col), dist = knn_edges(coords, k)
    zc = z - z.mean()
    w = rbf(dist, lengthscale)
    return float(len(z) / w.sum() * (w * zc[row] * zc[col]).sum() / (zc * zc).sum())


@torch.no_grad()
def gp_features(coords: torch.Tensor, d: int, lengthscale: float, mix_rank: int, jitter: float = 1e-5):
    """``d`` GP draws (RBF kernel) over ``coords``, mixed by a random rank-``mix_rank`` matrix, standardised."""
    K = torch.exp(-torch.cdist(coords, coords) ** 2 / (2.0 * lengthscale**2)) + jitter * torch.eye(len(coords))
    X = torch.linalg.cholesky(K) @ torch.randn(len(coords), d)
    X = X @ (torch.randn(d, mix_rank) @ torch.randn(d, mix_rank).T)
    return (X - X.mean(0)) / (X.std(0, unbiased=False) + 1e-8)


def make_sample(cfg: dict, content_direction: Optional[torch.Tensor] = None) -> RegressionGraph:
    n = random.randint(cfg["min_nodes"], cfg["max_nodes"])
    coords = 2 * torch.rand(n, 2) - 1
    a, b = math.log(cfg["gp_lengthscale"][0]), math.log(cfg["gp_lengthscale"][1])
    ell = math.exp(a + random.random() * (b - a))
    X = gp_features(coords, cfg["x_dim"], ell, cfg["mix_rank"])
    w = torch.randn(cfg["x_dim"])
    z = X @ (w / (w.norm() + 1e-12))
    z = (z - z.mean()) / (z.std(unbiased=False) + 1e-8)
    y_spatial = morans_i(coords, z, cfg["k_true"], cfg["weight_lengthscale"])
    y_feat = float(((X @ content_direction) ** 2).mean()) if content_direction is not None else 0.0

    n_ = len(coords)
    D = torch.cdist(coords, coords)
    off = ~torch.eye(n_, dtype=torch.bool)
    row, col = torch.nonzero(off, as_tuple=True)
    dist = D[off]
    graph = sail.build_graph(coords, X, method="dense", log_distances=False)
    return RegressionGraph(graph=graph, coords=coords, edge_index_fc=torch.stack([row, col]),
                           edge_weight_fc=rbf(dist, cfg["weight_lengthscale"]), edge_attr_fc=dist.view(-1, 1),
                           y_spatial=y_spatial, y_feat=y_feat, y=torch.tensor([y_spatial]))


def make_splits(cfg: dict, seed: int, content_direction: Optional[torch.Tensor] = None):
    """Train / validation / test sets for one seed."""
    sail.set_seed(seed)
    return [[make_sample(cfg, content_direction) for _ in range(cfg[f"n_{s}"])] for s in ("train", "val", "test")]


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class SAILRegressor(nn.Module):
    """SAIL (Moran-style, dense affinity, learned temperature) -> linear head."""

    def __init__(self, in_dim: int, n_heads: int) -> None:
        super().__init__()
        # sample-std standardisation, as in the run reported in the paper
        self.net = SAILPredictor(in_dim, n_heads=n_heads, n_outputs=1, local_index="moran",
                                 query_key_dim=None, standardise="sample")

    def forward(self, s: RegressionGraph) -> torch.Tensor:
        return self.net(s.graph.to(next(self.parameters()).device))[0]


class GCNRegressor(nn.Module):
    """One GCN layer on the fully connected RBF-weighted graph, mean pool, linear head."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv, self.head = GCNConv(in_dim, hidden), nn.Linear(hidden, 1)

    def forward(self, s: RegressionGraph) -> torch.Tensor:
        dev = next(self.parameters()).device
        h = F.relu(self.conv(s.x_baseline.to(dev), s.edge_index_fc.to(dev), edge_weight=s.edge_weight_fc.to(dev)))
        return self.head(h.mean(0, keepdim=True))


class GATRegressor(nn.Module):
    """One GATv2 layer with distance edge attributes on the fully connected graph."""

    def __init__(self, in_dim: int, hidden: int, heads: int) -> None:
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.conv = GATv2Conv(in_dim, hidden, heads=heads, concat=False, edge_dim=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, s: RegressionGraph) -> torch.Tensor:
        dev = next(self.parameters()).device
        h = F.relu(self.conv(s.x_baseline.to(dev), s.edge_index_fc.to(dev), edge_attr=s.edge_attr_fc.to(dev)))
        return self.head(h.mean(0, keepdim=True))


class MLPRegressor(nn.Module):
    """Node-wise MLP, mean pool, linear head (no graph)."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp, self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU()), nn.Linear(hidden, 1)

    def forward(self, s: RegressionGraph) -> torch.Tensor:
        return self.head(self.mlp(s.x_baseline.to(next(self.parameters()).device)).mean(0, keepdim=True))


@dataclass
class ModelSpec:
    build: Callable[[dict, int], nn.Module]
    use_coords: bool
    grid: list[dict]


def model_specs(hidden_grid: list[int], heads_grid: list[int]) -> dict[str, ModelSpec]:
    """Models of Fig. 3 / Supp. Table II with their hyper-parameter grids (smallest first)."""
    h = [dict(hidden=v) for v in hidden_grid]
    hh = [dict(hidden=v, heads=k) for v in hidden_grid for k in heads_grid]
    return {
        "SAIL (dense)": ModelSpec(lambda c, d: SAILRegressor(d, c["hidden"]), False, h),
        "GCN (X, FC)": ModelSpec(lambda c, d: GCNRegressor(d, c["hidden"]), False, h),
        "GCN (X+coords, FC)": ModelSpec(lambda c, d: GCNRegressor(d + 2, c["hidden"]), True, h),
        "GAT (X, FC)": ModelSpec(lambda c, d: GATRegressor(d, c["hidden"], c["heads"]), False, hh),
        "GAT (X+coords, FC)": ModelSpec(lambda c, d: GATRegressor(d + 2, c["hidden"], c["heads"]), True, hh),
        "MLP (X)": ModelSpec(lambda c, d: MLPRegressor(d, c["hidden"]), False, h),
        "MLP (X+coords)": ModelSpec(lambda c, d: MLPRegressor(d + 2, c["hidden"]), True, h),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if den <= 1e-12 else float((a * b).sum() / den)


@torch.no_grad()
def evaluate(model: nn.Module, data: list[RegressionGraph]) -> dict:
    model.eval()
    y = np.array([float(s.y[0]) for s in data])
    yhat = np.array([float(model(s).view(-1)[0]) for s in data])
    from scipy.stats import spearmanr

    return dict(mse=float(np.mean((yhat - y) ** 2)), pearson=pearson(yhat, y),
                spearman=float(spearmanr(yhat, y).statistic))


def fit(model: nn.Module, train, val, test, use_coords: bool, lr: float, max_epochs: int, patience: int,
        device: str = "cpu") -> dict:
    """Adam + MSE, one graph per step; early stopping on validation Pearson."""
    for s in (*train, *val, *test):
        s.x_baseline = torch.cat([s.x, s.coords], 1) if use_coords else s.x
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for s in train:
            opt.zero_grad()
            F.mse_loss(model(s), s.y.to(device).view(1, 1)).backward()
            opt.step()
        score = evaluate(model, val)["pearson"]
        if score > best + 1e-12:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return dict(val=evaluate(model, val), test=evaluate(model, test),
                params=sum(p.numel() for p in model.parameters() if p.requires_grad))
