"""Turn common spatial data containers into :class:`~sail.SpatialGraph` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .graph import SpatialGraph, build_graph


def anndata_to_graph(
    adata,
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    **graph_kwargs: Any,
) -> SpatialGraph:
    """Build a graph from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Cells/spots x features, coordinates in ``adata.obsm[spatial_key]``.
    spatial_key : str
        Key of the coordinate array in ``obsm``.
    layer : str, optional
        Use ``adata.layers[layer]`` instead of ``adata.X``.
    features : sequence of str, optional
        Subset of ``var_names`` to use as node features.
    **graph_kwargs
        Passed to :func:`sail.build_graph` (``method``, ``k``, ``radius``, ...).

    The feature names are stored in ``graph.meta["feature_names"]``.
    """
    if features is not None:
        adata = adata[:, list(features)]
    X = adata.layers[layer] if layer is not None else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    meta = {"feature_names": list(map(str, adata.var_names)), "obs_names": list(map(str, adata.obs_names))}
    return build_graph(np.asarray(adata.obsm[spatial_key]), X, meta=meta, **graph_kwargs)


def table_to_graphs(
    table: pd.DataFrame,
    coord_cols: Sequence[str],
    feature_cols: Sequence[str],
    group_col: Optional[str] = None,
    **graph_kwargs: Any,
) -> list[SpatialGraph]:
    """Build one graph per group (e.g. per slide) from a long table of nodes.

    Parameters
    ----------
    table : DataFrame
        One row per node.
    coord_cols, feature_cols : sequence of str
        Coordinate and feature columns.
    group_col : str, optional
        Column identifying the graph each node belongs to; a single graph if ``None``.

    The group value is stored in ``graph.meta["id"]``.
    """
    groups = [(None, table)] if group_col is None else table.groupby(group_col, sort=True)
    graphs = []
    for gid, df in groups:
        graphs.append(build_graph(
            df[list(coord_cols)].to_numpy(np.float32), df[list(feature_cols)].to_numpy(np.float32),
            meta={"id": gid, "feature_names": list(feature_cols)}, **graph_kwargs,
        ))
    return graphs


def save_graphs(graphs: Sequence[SpatialGraph], path: str | Path) -> None:
    """Save graphs to a single ``.pt`` file."""
    torch.save([g.__dict__ for g in graphs], path)


def load_graphs(path: str | Path) -> list[SpatialGraph]:
    """Load graphs written by :func:`save_graphs`."""
    return [SpatialGraph(**d) for d in torch.load(path, weights_only=False)]


def from_pyg(data, kind: str = "knn", meta: Optional[dict] = None) -> SpatialGraph:
    """Convert a ``torch_geometric`` ``Data`` with ``x``/``neighbors``/``distances`` (pre-release cache format)."""
    return SpatialGraph(
        x=data.x.float(), neighbors=data.neighbors.long(), distances=data.distances.float(),
        coords=getattr(data, "coords", None), kind=kind, meta=dict(meta or {}),
    )


# --------------------------------------------------------------------------- #
# cohort sample files (written by datasets/<cohort>/prepare.py)
# --------------------------------------------------------------------------- #
def save_samples(
    path: str | Path,
    ids: Sequence[str],
    features: Sequence[np.ndarray],
    coords: Sequence[np.ndarray],
    feature_names: Sequence[str],
) -> None:
    """Save a cohort as one node table per sample (graphs are built at run time).

    Parameters
    ----------
    ids : sequence of str
        Sample (patient / slide) identifiers, matching ``clinical.csv``.
    features : sequence of ndarray, each (N_i, d)
        Pre-processed node features.
    coords : sequence of ndarray, each (N_i, c)
        Node coordinates.
    """
    if not (len(ids) == len(features) == len(coords)):
        raise ValueError("ids, features and coords must have the same length")
    torch.save({"ids": list(map(str, ids)),
                "features": [np.asarray(f, dtype=np.float32) for f in features],
                "coords": [np.asarray(c, dtype=np.float32) for c in coords],
                "feature_names": list(feature_names)}, path)


def pooled_zscore(features: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Z-score every feature over the nodes of all samples together (population std).

    Arrays are converted to float32 and standardised in place to bound memory.
    """
    features = [np.asarray(x, dtype=np.float32) for x in features]
    n = sum(len(x) for x in features)
    mu = sum(x.sum(0, dtype=np.float64) for x in features) / n
    var = sum(np.square(x - mu.astype(np.float32), dtype=np.float64).sum(0) for x in features) / n
    sd = np.maximum(np.sqrt(var), 1e-8)
    for x in features:
        x -= mu.astype(np.float32)
        x /= sd.astype(np.float32)
    return features


def load_samples(path: str | Path) -> dict:
    """Read a file written by :func:`save_samples`."""
    return torch.load(path, weights_only=False)


def samples_to_graphs(samples: dict, ids: Optional[Sequence[str]] = None, **graph_kwargs: Any) -> list[SpatialGraph]:
    """Build one graph per sample (optionally only ``ids``, in that order)."""
    index = {sid: i for i, sid in enumerate(samples["ids"])}
    order = [index[s] for s in (ids if ids is not None else samples["ids"])]
    return [build_graph(samples["coords"][i], samples["features"][i],
                        meta={"id": samples["ids"][i], "feature_names": samples["feature_names"]},
                        **graph_kwargs)
            for i in order]
