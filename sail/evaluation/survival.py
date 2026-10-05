"""Prognostic evaluation of global SAIL scores (Section V-D).

Every score is assessed by a median split into high/low groups, reporting the
concordance index, log-rank p-value, Cox hazard ratio (high vs low) and the
difference in restricted mean survival time (low minus high).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SurvivalResult:
    c_index: float
    c_index_oriented: float  # max(c, 1 - c)
    logrank_p: float
    hr: float
    hr_lo: float
    hr_hi: float
    delta_rmst: float  # RMST(low) - RMST(high)
    n_high: int
    n_low: int


def median_split(scores: np.ndarray) -> np.ndarray:
    """Boolean "high" mask: ``score >= median``."""
    return scores >= np.median(scores)


def rmst(
    time: np.ndarray,
    event: np.ndarray,
    tau: float,
    method: Literal["trapezoid", "step"] = "trapezoid",
) -> float:
    """Restricted mean survival time up to ``tau`` from the Kaplan-Meier curve.

    ``method="trapezoid"`` integrates the KM curve with the trapezoidal rule
    over its event times up to ``tau`` (as used for the paper tables);
    ``"step"`` integrates the right-continuous step function exactly up to ``tau``.
    """
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(time, event_observed=event)
    t = np.concatenate([[0.0], km.survival_function_.index.values])
    s = np.concatenate([[1.0], km.survival_function_["KM_estimate"].values])
    if method == "trapezoid":
        keep = t <= tau
        return float(np.trapz(s[keep], t[keep]))
    if method == "step":
        t_cut = np.clip(np.append(t, tau), None, tau)
        return float(np.sum(s * np.diff(t_cut)))
    raise ValueError(f"Unknown RMST method {method!r}")


def evaluate_survival(
    scores: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    tau: float,
    rmst_method: Literal["trapezoid", "step"] = "trapezoid",
) -> SurvivalResult:
    """Median-split survival analysis of one score."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    from lifelines.utils import concordance_index

    scores, time, event = map(np.asarray, (scores, time, event))
    high = median_split(scores)
    low = ~high

    lr = logrank_test(time[high], time[low], event_observed_A=event[high], event_observed_B=event[low])
    c = concordance_index(time, scores, event_observed=event)
    cox = CoxPHFitter().fit(
        pd.DataFrame({"high": high.astype(int), "time": time, "event": event}),
        duration_col="time", event_col="event",
    )
    hr = float(np.exp(cox.params_["high"]))
    lo, hi = np.exp(cox.confidence_intervals_.loc["high"]).to_numpy()
    d_rmst = rmst(time[low], event[low], tau, rmst_method) - rmst(time[high], event[high], tau, rmst_method)

    return SurvivalResult(
        c_index=float(c), c_index_oriented=float(max(c, 1 - c)), logrank_p=float(lr.p_value),
        hr=hr, hr_lo=float(lo), hr_hi=float(hi), delta_rmst=float(d_rmst),
        n_high=int(high.sum()), n_low=int(low.sum()),
    )


def survival_table(
    scores: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    tau: float,
    names: Optional[Sequence[str]] = None,
    rmst_method: Literal["trapezoid", "step"] = "trapezoid",
) -> pd.DataFrame:
    """:func:`evaluate_survival` for every column of ``scores`` (shape (n_patients, h)).

    Rows are named ``Mode 1 ... Mode h`` unless ``names`` is given.
    """
    scores = np.asarray(scores)
    names = names or [f"Mode {i + 1}" for i in range(scores.shape[1])]
    rows = [
        dict(name=n, **asdict(evaluate_survival(scores[:, i], time, event, tau, rmst_method)))
        for i, n in enumerate(names)
    ]
    return pd.DataFrame(rows)


def plot_km(scores, time, event, ax=None, labels=("High SAIL", "Low SAIL"), title=None):
    """Kaplan-Meier curves of the median-split groups."""
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    ax = ax or plt.gca()
    high = median_split(np.asarray(scores))
    time, event = np.asarray(time), np.asarray(event)
    for mask, label in [(high, labels[0]), (~high, labels[1])]:
        KaplanMeierFitter().fit(time[mask], event_observed=event[mask], label=label).plot_survival_function(ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    if title:
        ax.set_title(title)
    return ax
