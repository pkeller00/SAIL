"""Fixed inverse-distance kNN weights for the classical baselines.

Baselines use the same kNN neighbourhood as SAIL but fixed (not learned)
weights ``w_ij proportional to 1 / d_ij``, row-standardised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import KDTree


@dataclass
class KNNWeights:
    """Row-standardised inverse-distance kNN weights.

    Attributes
    ----------
    neighbors : ndarray (N, k) of int
    weights : ndarray (N, k), rows sum to one
    """

    neighbors: np.ndarray
    weights: np.ndarray

    @property
    def n(self) -> int:
        return self.neighbors.shape[0]

    def to_sparse(self, symmetric: bool = False) -> sparse.csr_matrix:
        """(N, N) sparse weight matrix; ``symmetric`` gives ``(W + W^T) / 2``."""
        n, k = self.neighbors.shape
        W = sparse.csr_matrix((self.weights.ravel(), (np.repeat(np.arange(n), k), self.neighbors.ravel())),
                              shape=(n, n))
        return (W + W.T) / 2 if symmetric else W

    def to_libpysal(self):
        """``libpysal.weights.W`` (row-standardised) for esda."""
        from libpysal.weights import W as PySALWeights

        nb = {i: self.neighbors[i].tolist() for i in range(self.n)}
        wt = {i: self.weights[i].tolist() for i in range(self.n)}
        W = PySALWeights(nb, wt, silence_warnings=True)
        W.transform = "r"
        return W


def inverse_distance_knn(
    coords: np.ndarray, k: int = 32, log_distances: bool = True, eps: float = 1e-8
) -> KNNWeights:
    """kNN weights ``1 / (d + eps)``, row-standardised.

    ``log_distances=True`` uses ``log1p(d)`` in place of ``d`` (non-negative and
    monotonic), mirroring SAIL's log-distance input.
    """
    dist, idx = KDTree(np.asarray(coords, dtype=np.float64)).query(coords, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    if log_distances:
        dist = np.log1p(dist)
    w = 1.0 / (dist + eps)
    return KNNWeights(idx, w / w.sum(axis=1, keepdims=True))


def inverse_distance_from_graph(neighbors: np.ndarray, distances: np.ndarray, eps: float = 1e-12) -> KNNWeights:
    """Row-standardised ``1 / d`` weights on an existing neighbour list (raw distances)."""
    w = 1.0 / np.clip(np.asarray(distances, dtype=np.float64), eps, None)
    return KNNWeights(np.asarray(neighbors), w / w.sum(axis=1, keepdims=True))
