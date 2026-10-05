"""Hotspot baseline (DeTomaso & Yosef, 2021) on SAIL's kNN neighbourhood.

Features with significant spatial autocorrelation are grouped into modules by
their pairwise local correlation; each module gets a per-node score. Graphs
are pooled into one block-diagonal neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

from .weights import KNNWeights


@dataclass
class HotspotResult:
    """Hotspot output.

    Attributes
    ----------
    autocorrelation : DataFrame
        Per-feature Hotspot autocorrelation (``C``, ``Z``, ``Pval``, ``FDR``).
    local_correlation_z : DataFrame
        Pairwise local-correlation Z-scores of the selected features.
    modules : Series
        Module id per selected feature (-1 = unassigned).
    module_ids : list of int
    scores : ndarray, shape (N_total, n_modules)
        Per-node module scores (rows follow the concatenated graphs).
    """

    autocorrelation: pd.DataFrame
    local_correlation_z: pd.DataFrame
    modules: pd.Series
    module_ids: list[int]
    scores: np.ndarray

    def members(self, module: int) -> list[str]:
        return self.modules.index[self.modules == module].tolist()


def run_hotspot(
    Xs: Sequence[np.ndarray],
    weights: Sequence[KNNWeights],
    feature_names: Sequence[str],
    min_genes: int = 3,
    fdr_autocorrelation: float = 0.05,
    fdr_modules: float = 0.05,
    jobs: int = 1,
) -> HotspotResult:
    """Run Hotspot (``model="none"``) on one or more graphs.

    Parameters
    ----------
    Xs : sequence of ndarray, each (N_g, F)
        Node features per graph; z-scored over all nodes here.
    weights : sequence of KNNWeights
        Per-graph neighbourhoods; offset and stacked into one block-diagonal graph.
    feature_names : sequence of str, length F
    min_genes : int
        Minimum features per module (Hotspot's default of 20 is too large for
        protein panels).
    fdr_autocorrelation, fdr_modules : float
        FDR thresholds for feature selection and module formation.
    """
    from hotspot import modules as hs_modules
    from hotspot.knn import make_weights_non_redundant
    from hotspot.local_stats import compute_hs
    from hotspot.local_stats_pairs import compute_hs_pairs_centered_cond

    names = list(feature_names)
    X = np.concatenate([np.asarray(x, dtype=np.float64) for x in Xs]).T  # (F, N)
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    n = X.shape[1]

    offsets = np.cumsum([0] + [w.n for w in weights[:-1]])
    nb = np.concatenate([w.neighbors + o for w, o in zip(weights, offsets)]).astype(np.int64)
    wt = make_weights_non_redundant(nb, np.concatenate([w.weights for w in weights]))

    nb_ns, wt_ns = SimpleNamespace(values=nb), SimpleNamespace(values=wt)
    umi = SimpleNamespace(values=np.ones(n))  # unused by model="none"

    autocorr = compute_hs(X, nb_ns, wt_ns, umi, "none", genes=pd.Index(names), centered=True, jobs=jobs)
    selected = [g for g in names if autocorr.loc[g, "FDR"] < fdr_autocorrelation]
    sel_idx = [names.index(g) for g in selected]
    counts = SimpleNamespace(values=X[sel_idx], index=pd.Index(selected))
    _, local_z = compute_hs_pairs_centered_cond(counts, nb_ns, wt_ns, umi, "none", jobs=jobs)

    modules, _ = hs_modules.compute_modules(
        local_z, min_gene_threshold=min_genes, fdr_threshold=fdr_modules, core_only=True
    )
    module_ids = sorted(int(m) for m in modules.unique() if m != -1)
    scores = np.column_stack([
        hs_modules.compute_scores(X[[names.index(g) for g in modules.index[modules == m]]],
                                  "none", umi.values, nb, wt)
        for m in module_ids
    ]) if module_ids else np.empty((n, 0))
    return HotspotResult(autocorr.sort_values("Z", ascending=False), local_z, modules, module_ids, scores)
