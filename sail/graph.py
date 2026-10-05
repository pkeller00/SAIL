"""Spatial graph container and construction.

SAIL operates on *padded neighbour tensors*: for every node ``i`` we store the
indices of up to ``K`` neighbours and the corresponding (optionally
log-transformed) distances. All graph types are reduced to this layout:

=========  ===============================  ==========================
``kind``   ``K``                            padding mask
=========  ===============================  ==========================
``knn``    ``k``                            none (every slot is valid)
``radius`` max. neighbours within ``r``     padded slots are masked
``dense``  ``N - 1`` (all other nodes)      none
=========  ===============================  ==========================

so a single, vectorised forward pass (:class:`sail.SAIL`) serves every graph
type.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import Tensor

GraphKind = Literal["knn", "radius", "dense"]
Metric = Literal["euclidean", "haversine"]



@dataclass
class SpatialGraph:
    """A single spatial graph in padded-neighbour layout.

    Attributes
    ----------
    x : Tensor, shape (N, d)
        Node features.
    neighbors : LongTensor, shape (N, K)
        Neighbour indices of every node. Padded slots (see ``mask``) may hold
        any valid index; they receive zero affinity.
    distances : Tensor, shape (N, K)
        Distances to the neighbours in ``neighbors`` (log-transformed if the
        graph was built with ``log_distances=True``).
    mask : BoolTensor, shape (N, K), optional
        ``True`` for real neighbours, ``False`` for padding. ``None`` means
        every slot is valid.
    coords : Tensor, shape (N, c), optional
        Node coordinates (used for plotting only).
    kind : {"knn", "radius", "dense"}
        How the neighbourhood was constructed.
    meta : dict
        Free-form metadata (e.g. sample id, feature names).
    """

    x: Tensor
    neighbors: Tensor
    distances: Tensor
    mask: Optional[Tensor] = None
    coords: Optional[Tensor] = None
    kind: GraphKind = "knn"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.neighbors.shape != self.distances.shape:
            raise ValueError(
                f"neighbors {tuple(self.neighbors.shape)} and distances "
                f"{tuple(self.distances.shape)} must have the same shape"
            )
        if self.neighbors.shape[0] != self.x.shape[0]:
            raise ValueError("neighbors must have one row per node")
        if self.mask is not None and self.mask.shape != self.neighbors.shape:
            raise ValueError("mask must have the same shape as neighbors")
        self.neighbors = self.neighbors.long()

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    def to(self, device: torch.device | str) -> "SpatialGraph":
        """Return a copy with all tensors moved to ``device``."""

        def mv(t: Optional[Tensor]) -> Optional[Tensor]:
            return None if t is None else t.to(device)

        return SpatialGraph(
            x=self.x.to(device),
            neighbors=self.neighbors.to(device),
            distances=self.distances.to(device),
            mask=mv(self.mask),
            coords=mv(self.coords),
            kind=self.kind,
            meta=self.meta,
        )


def build_graph(
    coords: np.ndarray | Tensor,
    x: np.ndarray | Tensor,
    method: GraphKind = "knn",
    k: int = 32,
    radius: Optional[float] = None,
    metric: Metric = "euclidean",
    log_distances: bool = True,
    backend: Literal["auto", "sklearn", "cuvs"] = "auto",
    min_distance: float = 1e-6,
    meta: Optional[dict[str, Any]] = None,
) -> SpatialGraph:
    """Build a :class:`SpatialGraph` from node coordinates and features.

    Parameters
    ----------
    coords : array, shape (N, c)
        Node coordinates. For ``metric="haversine"`` these must be
        ``(latitude, longitude)`` in degrees; distances are then great-circle
        angles in radians (multiply by the Earth radius, 6371 km, for lengths).
    x : array, shape (N, d)
        Node features.
    method : {"knn", "radius", "dense"}
        Neighbourhood definition (see module docstring).
    k : int
        Number of neighbours for ``method="knn"`` (reduced to ``N - 1`` for
        smaller graphs, with a warning).
    radius : float, optional
        Neighbourhood radius for ``method="radius"`` (same units as the
        distance metric; radians for haversine).
    metric : {"euclidean", "haversine"}
        Distance metric.
    log_distances : bool
        If ``True`` (paper default) distances are replaced by
        ``log(max(d, min_distance))`` before being passed to the affinity.
    backend : {"auto", "sklearn", "cuvs"}
        kNN search backend. ``"auto"`` uses cuVS when installed and a GPU is
        available, otherwise scikit-learn. Only affects ``method="knn"`` with
        the Euclidean metric.
    min_distance : float
        Clamp applied before the log transform.
    meta : dict, optional
        Stored on the returned graph.

    Returns
    -------
    SpatialGraph
    """
    coords_np = _to_numpy(coords).astype(np.float32)
    x_t = torch.as_tensor(_to_numpy(x), dtype=torch.float32)
    n = coords_np.shape[0]
    mask = None

    if method == "knn":
        if k >= n:
            if n < 2:
                raise ValueError("a kNN graph needs at least two nodes")
            warnings.warn(f"k={k} >= number of nodes ({n}); using k={n - 1}", stacklevel=2)
            k = n - 1
        nbr, dist = _knn(coords_np, k, metric, backend)
    elif method == "radius":
        if radius is None:
            raise ValueError("method='radius' requires `radius`")
        nbr, dist, mask = _radius(coords_np, radius, metric)
    elif method == "dense":
        nbr, dist = _dense(coords_np, metric)
    else:
        raise ValueError(f"Unknown graph method: {method!r}")

    neighbors = torch.as_tensor(nbr, dtype=torch.long)
    distances = torch.as_tensor(dist, dtype=torch.float32)
    if log_distances:
        distances = torch.log(torch.clamp(distances, min=min_distance))
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool)

    return SpatialGraph(
        x=x_t,
        neighbors=neighbors,
        distances=distances,
        mask=mask,
        coords=torch.as_tensor(coords_np),
        kind=method,
        meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------- #
# Neighbour search back-ends
# --------------------------------------------------------------------------- #
def _to_numpy(a: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(a, Tensor):
        return a.detach().cpu().numpy()
    if hasattr(a, "toarray"):  # scipy sparse
        return np.asarray(a.toarray())
    return np.asarray(a)


def _sklearn_tree(coords: np.ndarray, metric: Metric):
    from sklearn.neighbors import BallTree, KDTree

    if metric == "haversine":
        return BallTree(np.radians(coords), metric="haversine")
    return KDTree(coords)


def _drop_self(nbr: np.ndarray, dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove each node from its own neighbour list (robust to duplicate points)."""
    n = nbr.shape[0]
    is_self = nbr == np.arange(n)[:, None]
    # nodes whose self-match was pushed out by duplicates: drop the last column
    no_self = ~is_self.any(axis=1)
    is_self[no_self, -1] = True
    keep = ~is_self
    return nbr[keep].reshape(n, k), dist[keep].reshape(n, k)


def _knn(coords: np.ndarray, k: int, metric: Metric, backend: str) -> tuple[np.ndarray, np.ndarray]:
    use_cuvs = metric == "euclidean" and backend in ("auto", "cuvs") and _cuvs_available()
    if backend == "cuvs" and not use_cuvs:
        raise RuntimeError("backend='cuvs' requested but cuVS / CUDA is not available")
    if use_cuvs:
        nbr, dist = _knn_cuvs(coords, k)
    else:
        query = np.radians(coords) if metric == "haversine" else coords
        dist, nbr = _sklearn_tree(coords, metric).query(query, k=k + 1)
    return _drop_self(nbr, dist, k)


def _cuvs_available() -> bool:
    try:
        import cupy  # noqa: F401
        from cuvs.neighbors import brute_force  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def _knn_cuvs(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp
    from cuvs.neighbors import brute_force

    coords_cu = cp.asarray(coords, dtype=cp.float32)
    index = brute_force.build(coords_cu, metric="euclidean")
    dist_cu, nbr_cu = brute_force.search(index, coords_cu, k + 1)
    return cp.asnumpy(cp.asarray(nbr_cu)), cp.asnumpy(cp.asarray(dist_cu))


def _radius(
    coords: np.ndarray, radius: float, metric: Metric
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.radians(coords) if metric == "haversine" else coords
    nbrs, dists = _sklearn_tree(coords, metric).query_radius(query, r=radius, return_distance=True)
    n = coords.shape[0]
    lists = []
    for i, (nb, d) in enumerate(zip(nbrs, dists)):
        keep = nb != i
        lists.append((nb[keep], d[keep]))
    width = max(1, max(len(nb) for nb, _ in lists))
    nbr = np.zeros((n, width), dtype=np.int64)
    dist = np.ones((n, width), dtype=np.float32)
    mask = np.zeros((n, width), dtype=bool)
    for i, (nb, d) in enumerate(lists):
        m = len(nb)
        nbr[i, :m], dist[i, :m], mask[i, :m] = nb, d, True
        nbr[i, m:] = i  # padding points at self; masked out in the affinity
    return nbr, dist, mask


def _dense(coords: np.ndarray, metric: Metric) -> tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    if metric == "haversine":
        from sklearn.metrics.pairwise import haversine_distances

        full = haversine_distances(np.radians(coords))
    else:
        diff = coords[:, None, :] - coords[None, :, :]
        full = np.sqrt((diff**2).sum(-1))
    others = ~np.eye(n, dtype=bool)
    nbr = np.broadcast_to(np.arange(n), (n, n))[others].reshape(n, n - 1)
    dist = full[others].reshape(n, n - 1)
    return nbr, dist


def concat_graphs(graphs: list[SpatialGraph], meta: Optional[dict[str, Any]] = None) -> SpatialGraph:
    """Merge graphs into one block-diagonal graph (no edges between the parts).

    Used when one sample consists of several disconnected pieces of tissue
    (e.g. all tissue cores of a patient). Neighbour indices are offset;
    neighbourhood widths must agree. Each part's node range is stored in
    ``meta["parts"]`` as ``(start, stop)``.
    """
    widths = {g.neighbors.shape[1] for g in graphs}
    if len(widths) != 1:
        raise ValueError(f"all parts need the same neighbourhood width, got {sorted(widths)}")
    offsets = np.cumsum([0] + [g.num_nodes for g in graphs])
    has_mask = any(g.mask is not None for g in graphs)
    has_coords = all(g.coords is not None for g in graphs)
    return SpatialGraph(
        x=torch.cat([g.x for g in graphs]),
        neighbors=torch.cat([g.neighbors + int(o) for g, o in zip(graphs, offsets)]),
        distances=torch.cat([g.distances for g in graphs]),
        mask=torch.cat([g.mask if g.mask is not None else torch.ones_like(g.neighbors, dtype=torch.bool)
                        for g in graphs]) if has_mask else None,
        coords=torch.cat([g.coords for g in graphs]) if has_coords else None,
        kind=graphs[0].kind,
        meta={**(meta or {}), "parts": [(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:])]},
    )
