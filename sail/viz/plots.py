"""Mode-level figures: loadings and LISA maps for every learned mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..graph import SpatialGraph
from ..lisa import lisa
from ..modes import SAILModes
from .images import ImageSource
from .overlay import overlay
from .style import NEG_WEIGHT_COLOR, POS_WEIGHT_COLOR, lisa_legend_handles


def plot_mode_weights(
    W, feature_names: Sequence[str], mode: int, ax: Optional[plt.Axes] = None, top_k: int = 10,
    title: Optional[str] = None,
) -> plt.Axes:
    """Bar chart of the ``top_k`` largest-magnitude loadings of one mode.

    Parameters
    ----------
    W : array, shape (d, h)
        Mode loadings (``SAILModes.W``).
    feature_names : sequence of str, length d
    mode : int
        Zero-based mode index.
    """
    w = np.asarray(W.detach().cpu() if hasattr(W, "detach") else W)[:, mode]
    top = np.argsort(-np.abs(w))[:top_k]
    ax = ax or plt.gca()
    ax.bar(range(len(top)), w[top], color=[POS_WEIGHT_COLOR if x >= 0 else NEG_WEIGHT_COLOR for x in w[top]])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([feature_names[i] for i in top], rotation=60, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(-0.5, len(top) - 0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def plot_modes(
    graph: SpatialGraph,
    modes: SAILModes,
    feature_names: Sequence[str],
    lisa_labels: Optional[np.ndarray] = None,
    image: Optional[ImageSource] = None,
    which: Optional[Sequence[int]] = None,
    ncols: int = 4,
    top_k: int = 10,
    point_size: float = 4.0,
    panel_size: float = 4.0,
    save_path: Optional[str | Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """LISA map and loading bar chart for each mode (as in Supp. Fig. 5).

    Parameters
    ----------
    graph : SpatialGraph
        Graph whose nodes are drawn (needs ``coords``).
    modes : SAILModes
        Trained modes.
    feature_names : sequence of str
        Names of the input features (bar labels).
    lisa_labels : ndarray, shape (N, h), optional
        Pre-computed :func:`sail.lisa` output; computed if ``None``.
    image : ImageSource, optional
        Background for the spatial panels.
    which : sequence of int, optional
        Zero-based modes to plot (default: all).
    """
    h = modes.W.shape[1]
    which = list(range(h)) if which is None else list(which)
    if lisa_labels is None:
        lisa_labels = lisa(graph, modes.model, modes=which)
        col_of = {m: i for i, m in enumerate(which)}
    else:
        col_of = {m: m for m in which}

    nrows = int(np.ceil(len(which) / ncols))
    fig, axs = plt.subplots(2 * nrows, ncols, figsize=(ncols * panel_size, 2 * nrows * panel_size), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")

    scores = modes.mean_scores
    for i, m in enumerate(which):
        r, c = divmod(i, ncols)
        ax_map, ax_bar = axs[2 * r, c], axs[2 * r + 1, c]
        label = f"Mode {m + 1}" + (f"  (score {float(scores[m]):.3f})" if scores is not None else "")
        overlay(graph.coords, lisa_labels[:, col_of[m]], image=image, kind="lisa", ax=ax_map,
                point_size=point_size, title=label, legend=False)
        ax_bar.axis("on")
        plot_mode_weights(modes.W, feature_names, m, ax=ax_bar, top_k=top_k)

    fig.legend(handles=lisa_legend_handles(), title="LISA cluster", loc="lower left",
               ncol=5, frameon=False, fontsize=9)
    fig.legend(handles=[Patch(color=POS_WEIGHT_COLOR, label="positive loading"),
                        Patch(color=NEG_WEIGHT_COLOR, label="negative loading")],
               title="Mode weights", loc="lower right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
