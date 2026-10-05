"""Plot node values over their spatial layout, optionally on a background image."""

from __future__ import annotations

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .images import ImageSource
from .style import LISA_COLORS, lisa_legend_handles


def _numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def overlay(
    coords,
    values,
    image: Optional[ImageSource] = None,
    kind: Literal["continuous", "lisa"] = "continuous",
    ax: Optional[plt.Axes] = None,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    point_size: float = 4.0,
    nonsig_size: Optional[float] = None,
    alpha: float = 0.8,
    title: Optional[str] = None,
    colorbar: bool = True,
    legend: bool = True,
) -> plt.Axes:
    """Scatter one value per node at its location.

    Parameters
    ----------
    coords : array, shape (N, 2) or (N, 3)
        Node coordinates (e.g. ``graph.coords``). 3-D coordinates are drawn
        on a 3-D axis (without a background image).
    values : array, shape (N,)
        A continuous field (local SAIL score, ``f_map(x)``) or LISA label ids.
    image : ImageSource, optional
        Background image (:class:`~sail.viz.AnnDataImage`,
        :class:`~sail.viz.WSIImage`, :class:`~sail.viz.ArrayImage`). Points are
        mapped to its pixel grid with ``image.to_pixels``.
    kind : {"continuous", "lisa"}
        ``"lisa"`` colours by LISA category and shrinks non-significant nodes.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if ``None``.
    cmap, vmin, vmax
        Colour map and limits for continuous values.
    point_size : float
        Marker size (significant nodes when ``kind="lisa"``).
    nonsig_size : float, optional
        Marker size of non-significant nodes (default ``point_size / 3``).

    Returns
    -------
    matplotlib Axes

    Notes
    -----
    Image convention is used throughout: ``y`` grows downwards, with or without
    a background image.
    """
    xy = _numpy(coords).astype(float)
    v = _numpy(values).reshape(-1)
    if len(v) != len(xy):
        raise ValueError(f"{len(v)} values for {len(xy)} nodes")
    is_3d = xy.shape[1] == 3 and image is None

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)

    if image is not None:
        ax.imshow(image.image(), origin="upper")
        xy = image.to_pixels(xy)

    pts = xy.T if is_3d else (xy[:, 0], xy[:, 1])
    if kind == "lisa":
        ids = v.astype(int)
        colors = np.array([LISA_COLORS.get(i, "#000000") for i in ids])
        sizes = np.where(ids == 0, point_size / 3 if nonsig_size is None else nonsig_size, point_size)
        order = np.argsort(ids != 0)  # draw significant nodes on top
        ax.scatter(*[p[order] for p in pts], c=colors[order], s=sizes[order], alpha=alpha, linewidths=0)
        if legend:
            present = [i for i in (1, 2, 3, 4, 0) if i in set(ids)]
            ax.legend(handles=lisa_legend_handles(present), loc="best", frameon=False, fontsize=8)
    elif kind == "continuous":
        sc = ax.scatter(*pts, c=v, cmap=cmap, vmin=vmin, vmax=vmax, s=point_size, alpha=alpha, linewidths=0)
        if colorbar:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    else:
        raise ValueError(f"Unknown kind {kind!r}")

    if not is_3d:
        ax.set_aspect("equal")
        if image is None and not ax.yaxis_inverted():
            ax.invert_yaxis()
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
