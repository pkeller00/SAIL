"""Classical (fixed-weight) spatial autocorrelation baselines.

* Global Moran's I / Geary's C per feature, and their mean over features
  (the "Univariate" and "Mean" baselines of the survival tables).
* Local Moran's I LISA maps on a single feature or on the composite of all
  features (the "Univariate" and "Mean" baselines of the 4i Dice table).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .weights import KNNWeights

#: esda ``Moran_Local.q`` quadrant (1=HH, 2=LH, 3=LL, 4=HL) -> :data:`sail.LISA_LABELS` id.
_ESDA_Q_TO_LISA = np.array([0, 1, 3, 2, 4])


def global_moran_geary(
    X: np.ndarray, neighbors: np.ndarray, distances: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Global Moran's I and Geary's C of every column of ``X`` on a kNN graph.

    Weights are inverse distances ``1 / d_ij``; each edge is added in both
    directions (symmetrised) and the result is row-standardised.

    Parameters
    ----------
    X : ndarray, shape (N, F)
    neighbors : ndarray, shape (N, k)
    distances : ndarray, shape (N, k)
        Raw (not log-transformed) neighbour distances.

    Returns
    -------
    moran, geary : ndarray, shape (F,)
    """
    X = np.asarray(X, dtype=np.float64)
    n, k = neighbors.shape
    i = np.repeat(np.arange(n), k)
    j = np.asarray(neighbors).ravel().astype(np.int64)
    keep = i != j
    i, j = i[keep], j[keep]
    v = 1.0 / np.clip(np.asarray(distances, dtype=np.float64).ravel()[keep], eps, None)
    i, j, v = np.concatenate([i, j]), np.concatenate([j, i]), np.concatenate([v, v])
    v = v / np.clip(np.bincount(i, weights=v, minlength=n)[i], eps, None)

    S0 = v.sum()
    z = X - X.mean(0)
    ss = np.clip((z**2).sum(0), eps, None)
    moran = n / S0 * (v[:, None] * z[i] * z[j]).sum(0) / ss
    geary = (n - 1) * (v[:, None] * (X[i] - X[j]) ** 2).sum(0) / (2 * S0 * ss)
    return moran, geary


def composite_mean(X: np.ndarray) -> np.ndarray:
    """Mean of the z-scored columns (each feature contributes equally); constant columns are ignored."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(0)
    Z = np.where(sd > 1e-12, (X - X.mean(0)) / np.where(sd > 1e-12, sd, 1), 0.0)
    return Z.mean(1)


def local_moran_lisa(
    values: np.ndarray,
    w: KNNWeights,
    permutations: int = 999,
    alpha: float = 0.05,
    test: Literal["two_sided_bh", "one_sided_raw"] = "two_sided_bh",
    seed: int | None = None,
) -> np.ndarray:
    """Local Moran's I LISA labels (:data:`sail.LISA_LABELS` ids) using esda.

    Parameters
    ----------
    values : ndarray, shape (N,)
    w : KNNWeights
    permutations : int
        Conditional permutations (esda).
    test : {"two_sided_bh", "one_sided_raw"}
        ``"two_sided_bh"``: two-sided pseudo p (``2 * p_sim``) with
        Benjamini-Hochberg correction across nodes, as for SAIL.
        ``"one_sided_raw"``: esda's folded pseudo p ``p_sim`` without correction
        (used for the univariate 4i baseline in the paper).
    seed : int, optional
        esda permutation seed (``None`` uses NumPy's global RNG, as in the paper runs).
    """
    from esda.moran import Moran_Local
    from statsmodels.stats.multitest import multipletests

    ml = Moran_Local(np.asarray(values, dtype=np.float64), w.to_libpysal(),
                     permutations=permutations, seed=seed)
    quadrant = _ESDA_Q_TO_LISA[ml.q]
    if test == "two_sided_bh":
        p = multipletests(np.minimum(2.0 * np.asarray(ml.p_sim), 1.0), method="fdr_bh")[1]
    elif test == "one_sided_raw":
        p = np.asarray(ml.p_sim)
    else:
        raise ValueError(f"Unknown test {test!r}")
    return np.where(p < alpha, quadrant, 0).astype(np.int64)
