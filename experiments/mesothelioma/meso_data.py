"""MesoGraph St George cohort: graph loading, shared CV splits and metrics.

The per-core graphs are the ones distributed with MesoGraph (Eastwood et al.,
Cell Reports Medicine 2023): ``<core>.pkl`` torch_geometric objects with 617
cell features (morphology, intensity, texture, ResNet) and a trailing virtual
"core node" that only MesoGraph uses. Labels: 0 = epithelioid, 1 = biphasic,
2 = sarcomatoid; the task is epithelioid vs non-epithelioid.
"""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import sail


def _load_cpu(b):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


# The distributed graphs were pickled on a GPU; force CPU storage on load.
torch.storage._load_from_bytes = _load_cpu


def load_cores(graph_dir: Path) -> dict:
    """Raw MesoGraph graphs keyed by core id (e.g. ``"13-A"``)."""
    cores = {}
    for f in sorted(Path(graph_dir).glob("*.pkl")):
        with open(f, "rb") as fh:
            g = pickle.load(fh)
        cores[str(getattr(g, "core", f.stem))] = g
    return cores


def label(g) -> int:
    """3-class core label (0 E, 1 B, 2 S)."""
    return int(np.ravel(g.y)[0])


def binarize(y3: int) -> int:
    """Epithelioid (0) vs non-epithelioid (1)."""
    return int(y3 >= 1)


def to_sail_graph(g, graph_kwargs: dict) -> sail.SpatialGraph:
    """Cell graph for SAIL: real cells only (virtual core node dropped), kNN from centroids."""
    x = torch.as_tensor(np.asarray(g.x.detach().cpu() if torch.is_tensor(g.x) else g.x), dtype=torch.float32)[:-1]
    xy = torch.as_tensor(np.asarray(g.coords.detach().cpu() if torch.is_tensor(g.coords) else g.coords),
                         dtype=torch.float32)[:-1]
    return sail.build_graph(xy, x, **graph_kwargs)


def tma_slide(core: str) -> int:
    """Physical TMA slide of a St George core (MesoGraph ``utils.map_ind``)."""
    block = int(core.split("-")[0])
    if not 3 <= block <= 44:
        raise ValueError(f"{core} is not a St George core")
    return int(np.searchsorted([13, 25, 37], block, side="right"))


def make_splits(cores: dict, val_frac: float, seed: int) -> dict:
    """Leave-one-slide-out folds with a stratified 75/25 inner train/val split."""
    from sklearn.model_selection import train_test_split

    order = sorted(cores)
    y3 = np.array([label(cores[c]) for c in order])
    slide = np.array([tma_slide(c) for c in order])
    folds = []
    for f, s in enumerate(np.unique(slide)):
        tr, te = np.flatnonzero(slide != s), np.flatnonzero(slide == s)
        tr_idx, va_idx = train_test_split(tr, test_size=val_frac, shuffle=True, stratify=y3[tr], random_state=seed)
        folds.append(dict(fold=f, slide=int(s), train=[order[i] for i in tr_idx],
                          val=[order[i] for i in va_idx], test=[order[i] for i in te]))
    return dict(seed=seed, val_frac=val_frac, n_folds=len(folds), cores=order,
                labels={c: int(y) for c, y in zip(order, y3)}, folds=folds)


def load_splits(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def auc_ap(y_bin, score) -> tuple[float, float]:
    """ROC-AUC and average precision (NaN when only one class is present)."""
    y_bin, score = np.asarray(y_bin), np.asarray(score)
    if len(np.unique(y_bin)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_bin, score)), float(average_precision_score(y_bin, score))
