"""Background images for overlays.

An *image source* provides a background image and the mapping from graph
coordinates to that image's pixel grid::

    pixels = (coords - offset) * scale

Three sources cover the common cases:

* :class:`ArrayImage`   - any in-memory image (numpy array).
* :class:`AnnDataImage` - Visium-style ``adata.uns["spatial"][library_id]["images"][img_key]``,
  scaled with the matching ``tissue_<img_key>_scalef`` scale factor.
* :class:`WSIImage`     - a whole-slide image read at reduced resolution with tiatoolbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np


class ImageSource(Protocol):
    """Anything that can provide a background image and a coordinate transform."""

    def image(self) -> np.ndarray:
        """Image array of shape (H, W) or (H, W, C)."""

    def to_pixels(self, coords: np.ndarray) -> np.ndarray:
        """Map graph coordinates (N, 2) to image pixel coordinates (N, 2) as ``(x, y)``."""


@dataclass
class ArrayImage:
    """An in-memory image.

    Parameters
    ----------
    array : ndarray, shape (H, W) or (H, W, C)
        The image.
    scale : float or (float, float)
        Pixels per coordinate unit (per axis).
    offset : (float, float)
        Coordinate that maps to pixel ``(0, 0)``.
    """

    array: np.ndarray
    scale: float | Sequence[float] = 1.0
    offset: Sequence[float] = (0.0, 0.0)

    def image(self) -> np.ndarray:
        return np.asarray(self.array)

    def to_pixels(self, coords: np.ndarray) -> np.ndarray:
        return (np.asarray(coords)[:, :2] - np.asarray(self.offset)) * np.asarray(self.scale)


class AnnDataImage:
    """An image stored in an AnnData object (Visium / squidpy layout).

    Parameters
    ----------
    adata : AnnData
        Object with ``uns[spatial_key][library_id]["images"][img_key]``.
    img_key : str
        Image key, e.g. ``"hires"``, ``"lowres"`` or any custom key.
    library_id : str, optional
        Defaults to the only library in ``uns[spatial_key]``.
    scale : float, optional
        Pixels per coordinate unit. Defaults to
        ``scalefactors["tissue_<img_key>_scalef"]`` (or ``1.0`` if absent).
    spatial_key : str
        Key in ``adata.uns`` holding the image dictionary.
    """

    def __init__(
        self,
        adata,
        img_key: str = "hires",
        library_id: Optional[str] = None,
        scale: Optional[float] = None,
        spatial_key: str = "spatial",
    ) -> None:
        libraries = adata.uns[spatial_key]
        if library_id is None:
            if len(libraries) != 1:
                raise ValueError(f"Several libraries in uns[{spatial_key!r}]; pass library_id from {list(libraries)}")
            library_id = next(iter(libraries))
        lib = libraries[library_id]
        if img_key not in lib["images"]:
            raise KeyError(f"img_key {img_key!r} not found; available: {list(lib['images'])}")
        self._image = np.asarray(lib["images"][img_key])
        if scale is None:
            scale = lib.get("scalefactors", {}).get(f"tissue_{img_key}_scalef", 1.0)
        self.scale = float(scale)

    def image(self) -> np.ndarray:
        return self._image

    def to_pixels(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords)[:, :2] * self.scale


class WSIImage:
    """A whole-slide image, read as a thumbnail through tiatoolbox.

    Parameters
    ----------
    path : str or Path
        Slide file.
    coord_scale : float or (float, float)
        Level-0 (baseline) pixels per graph-coordinate unit. For coordinates
        in microns this is ``1 / slide_mpp``; for coordinates already in
        level-0 pixels it is ``1``; for coordinates in pixels at a different
        resolution it is ``coord_mpp / slide_mpp``.
    offset : (float, float)
        Graph coordinate of the slide's level-0 origin.
    resolution, units
        Thumbnail resolution, passed to ``WSIReader.slide_thumbnail``
        (e.g. ``1.25, "power"`` or ``8, "mpp"``).
    """

    def __init__(
        self,
        path,
        coord_scale: float | Sequence[float] = 1.0,
        offset: Sequence[float] = (0.0, 0.0),
        resolution: float = 1.25,
        units: str = "power",
    ) -> None:
        try:
            from tiatoolbox.wsicore.wsireader import WSIReader
        except ImportError as err:  # pragma: no cover
            raise ImportError("WSIImage requires tiatoolbox (pip install sail[wsi])") from err
        reader = WSIReader.open(path)
        self._image = reader.slide_thumbnail(resolution=resolution, units=units)
        base_w, base_h = reader.slide_dimensions(resolution=0, units="level")
        thumb_h, thumb_w = self._image.shape[:2]
        self._baseline_to_thumb = np.array([thumb_w / base_w, thumb_h / base_h])
        self.coord_scale = np.asarray(coord_scale, dtype=float)
        self.offset = np.asarray(offset, dtype=float)

    def image(self) -> np.ndarray:
        return self._image

    def to_baseline(self, coords: np.ndarray) -> np.ndarray:
        """Graph coordinates -> level-0 slide pixels."""
        return (np.asarray(coords)[:, :2] - self.offset) * self.coord_scale

    def to_pixels(self, coords: np.ndarray) -> np.ndarray:
        return self.to_baseline(coords) * self._baseline_to_thumb
