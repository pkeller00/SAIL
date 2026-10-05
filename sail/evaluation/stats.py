"""Multiple-testing and paired-comparison helpers."""

from __future__ import annotations

import numpy as np


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaNs are passed through."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def paired_wilcoxon(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> float:
    """Paired Wilcoxon signed-rank p-value for ``a`` vs ``b`` (e.g. SAIL vs baseline Dice)."""
    from scipy.stats import wilcoxon

    return float(wilcoxon(a, b, alternative=alternative).pvalue)
