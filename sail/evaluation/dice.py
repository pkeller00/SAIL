"""Overlap between LISA hotspot masks and reference annotations."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..lisa import LISA_LABELS


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Dice coefficient ``2 |P & T| / (|P| + |T|)`` of two binary masks."""
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    return float(2 * np.sum(p & t) / (p.sum() + t.sum() + eps))


def lisa_region_dice(
    lisa_labels: np.ndarray,
    regions: Mapping[str, np.ndarray],
    clusters: Sequence[str] = ("High-High", "Low-Low"),
) -> pd.DataFrame:
    """Dice of every (mode, LISA cluster, region) combination.

    Parameters
    ----------
    lisa_labels : ndarray, shape (N, h)
        Output of :func:`sail.lisa`.
    regions : mapping name -> bool mask of shape (N,)
        Reference annotations.
    clusters : sequence of str
        LISA categories whose masks are compared to the regions.

    Returns
    -------
    DataFrame with columns ``mode`` (1-based), ``lisa_cluster``, ``region``, ``dice``.
    Empty hotspot masks are skipped.
    """
    rows = []
    for m in range(lisa_labels.shape[1]):
        for name in clusters:
            mask = lisa_labels[:, m] == LISA_LABELS[name]
            if not mask.any():
                continue
            for region, target in regions.items():
                rows.append(dict(mode=m + 1, lisa_cluster=name, region=region,
                                 dice=dice_score(mask, target)))
    return pd.DataFrame(rows, columns=["mode", "lisa_cluster", "region", "dice"])


def best_per_region(dice_table: pd.DataFrame) -> pd.DataFrame:
    """Best-matching (mode, cluster) per region, sorted by Dice (Supp. Table IV)."""
    idx = dice_table.groupby("region")["dice"].idxmax()
    return dice_table.loc[idx].sort_values("dice", ascending=False).reset_index(drop=True)
