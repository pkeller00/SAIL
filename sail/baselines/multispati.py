"""MULTISPATI-PCA baseline (Dray et al., 2008).

Axes ``a`` maximise ``variance(Xa) * MoranI(Xa)``, i.e. they are the leading
eigenvectors of ``H = sum_g X_g^T (W_g + W_g^T)/2 X_g / N`` pooled over all
graphs ``g`` (a block-diagonal graph), with uniform row weights and the
identity column metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .weights import KNNWeights


@dataclass
class MultispatiResult:
    """Fitted MULTISPATI-PCA.

    Attributes
    ----------
    loadings : ndarray, shape (F, F)
        Unit-norm axes as columns, ordered by decreasing eigenvalue.
    eigenvalues, variance, moran : ndarray, shape (F,)
        ``eigenvalue = variance * moran`` of each axis score.
    kept : ndarray of bool, shape (F,)
        Leading positive axes explaining ``eig_frac`` of the positive eigenvalue.
    mean, std : ndarray, shape (F,)
        Pooled standardisation used before projection.
    """

    loadings: np.ndarray
    eigenvalues: np.ndarray
    variance: np.ndarray
    moran: np.ndarray
    kept: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray, kept_only: bool = True) -> np.ndarray:
        """Axis scores of ``X`` (N, F) -> (N, n_axes)."""
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        L = self.loadings[:, self.kept] if kept_only else self.loadings
        return Z @ L

    def axes_table(self) -> pd.DataFrame:
        pos = np.clip(self.eigenvalues, 0, None)
        return pd.DataFrame({
            "axis": [f"MS{i + 1}" for i in range(len(self.eigenvalues))],
            "eigenvalue": self.eigenvalues,
            "variance": self.variance,
            "moran_i": self.moran,
            "cum_frac": np.cumsum(pos) / pos.sum(),
            "kept": self.kept,
        })


def multispati_pca(
    Xs: Sequence[np.ndarray], weights: Sequence[KNNWeights], eig_frac: float = 0.80
) -> MultispatiResult:
    """Fit MULTISPATI-PCA on one or more graphs.

    Parameters
    ----------
    Xs : sequence of ndarray, each (N_g, F)
        Node features per graph (standardised internally over all nodes).
    weights : sequence of KNNWeights
        Row-standardised spatial weights per graph (symmetrised internally).
    eig_frac : float
        Keep the leading axes until this fraction of the positive eigenvalue
        is reached (chosen before looking at any outcome).
    """
    Xs = [np.asarray(X, dtype=np.float64) for X in Xs]
    n_total = sum(len(X) for X in Xs)
    mean = sum(X.sum(0) for X in Xs) / n_total
    std = np.sqrt(sum((X**2).sum(0) for X in Xs) / n_total - mean**2)
    std = np.where(std > 1e-10, std, 1.0)

    F = Xs[0].shape[1]
    H, C = np.zeros((F, F)), np.zeros((F, F))
    for X, w in zip(Xs, weights):
        Z = (X - mean) / std
        ZtWZ = Z.T @ np.asarray(w.to_sparse() @ Z)
        H += (ZtWZ + ZtWZ.T) / 2
        C += Z.T @ Z
    H /= n_total
    C /= n_total

    eig, vec = np.linalg.eigh(H)
    order = np.argsort(eig)[::-1]
    eig, loadings = eig[order], vec[:, order]
    var = np.einsum("pi,pq,qi->i", loadings, C, loadings)

    pos = np.clip(eig, 0, None)
    n_keep = int(np.searchsorted(np.cumsum(pos) / pos.sum(), eig_frac) + 1)
    kept = (np.arange(F) < n_keep) & (eig > 0)
    return MultispatiResult(loadings, eig, var, eig / var, kept, mean, std)
