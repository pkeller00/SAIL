"""Association of SAIL scores with a binary label (e.g. treatment response)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score

from .stats import fdr_bh


def association_table(scores: pd.DataFrame, labels: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Per-score two-sided Mann-Whitney U test, ROC AUC and BH-FDR.

    Parameters
    ----------
    scores : DataFrame
        One row per unit (e.g. patient), one column per score.
    labels : array of {0, 1}
        Binary label; 1 is the positive group (e.g. responders).
    columns : sequence of str
        Score columns to test.

    Returns
    -------
    DataFrame with, per score, the group medians, ``auc`` (positive group
    scored higher), the orientation-free ``auc_flip = max(auc, 1 - auc)``,
    ``higher_in_positive``, the U statistic and raw / BH-adjusted p-values,
    sorted by p-value.
    """
    y = np.asarray(labels).astype(int)
    rows = []
    for col in columns:
        x = scores[col].to_numpy(float)
        pos, neg = x[y == 1], x[y == 0]
        u, p = mannwhitneyu(pos, neg, alternative="two-sided")
        auc = roc_auc_score(y, x)
        rows.append(dict(score=col, n_pos=len(pos), n_neg=len(neg),
                         median_pos=float(np.median(pos)), median_neg=float(np.median(neg)),
                         auc=float(auc), auc_flip=float(max(auc, 1 - auc)),
                         higher_in_positive=bool(auc >= 0.5), U=float(u), pvalue=float(p)))
    table = pd.DataFrame(rows)
    table["p_fdr_bh"] = fdr_bh(table["pvalue"].to_numpy())
    return table.sort_values("pvalue").reset_index(drop=True)


def bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    higher_in_positive: bool = True,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the AUC, resampling units with replacement.

    The direction of the score is fixed from the full data
    (``higher_in_positive``); resamples containing a single class are dropped.
    Ties are handled with average ranks, as in ``roc_auc_score``.
    """
    y = np.asarray(labels).astype(int)
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yb = y[idx]
    n1 = yb.sum(1)
    ok = (n1 > 0) & (n1 < len(y))
    yb, n1 = yb[ok], n1[ok]
    xb = (x if higher_in_positive else -x)[idx[ok]]
    ranks = rankdata(xb, axis=1)
    auc = ((ranks * yb).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(auc, [tail, 100 - tail])
    return float(lo), float(hi)
