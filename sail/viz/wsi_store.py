"""Export per-node SAIL outputs as a tiatoolbox annotation store for WSI viewers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ..lisa import LISA_NAMES


def export_annotation_store(
    coords: np.ndarray,
    path: str | Path,
    values: Mapping[str, Sequence[float]],
    lisa_labels: Optional[Sequence[int]] = None,
    coord_scale: float | Sequence[float] = 1.0,
    offset: Sequence[float] = (0.0, 0.0),
    radius: float = 20.0,
) -> Path:
    """Write one circular annotation per node to a tiatoolbox ``SQLiteStore``.

    The store can be opened in the tiatoolbox visualisation tool on top of
    the slide.

    Parameters
    ----------
    coords : array, shape (N, 2)
        Node coordinates in graph units.
    path : str or Path
        Output ``.db`` file.
    values : mapping name -> array of length N
        Properties stored on each annotation (e.g. ``{"mode_1": local[:, 0]}``).
    lisa_labels : array of int, length N, optional
        Stored as the ``type`` property (LISA category name).
    coord_scale, offset
        Map graph coordinates to level-0 slide pixels:
        ``(coords - offset) * coord_scale`` (see :class:`sail.viz.WSIImage`).
    radius : float
        Annotation radius in level-0 pixels.
    """
    try:
        from shapely.geometry import Point
        from tiatoolbox.annotation.storage import Annotation, SQLiteStore
    except ImportError as err:  # pragma: no cover
        raise ImportError("export_annotation_store requires tiatoolbox (pip install sail[wsi])") from err

    px = (np.asarray(coords)[:, :2] - np.asarray(offset)) * np.asarray(coord_scale)
    names = list(values)
    columns = [np.asarray(values[n], dtype=float) for n in names]
    annotations = []
    for i, (x, y) in enumerate(px):
        props = {n: float(col[i]) for n, col in zip(names, columns)}
        if lisa_labels is not None:
            props["type"] = LISA_NAMES[int(lisa_labels[i])]
        annotations.append(Annotation(Point(float(x), float(y)).buffer(radius), properties=props))

    store = SQLiteStore()
    store.append_many(annotations)
    path = Path(path)
    store.dump(path)
    return path
