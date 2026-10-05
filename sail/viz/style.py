"""Shared colours (PySAL / GeoDa LISA convention)."""

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..lisa import LISA_NAMES

LISA_COLORS: dict[int, str] = {
    0: "#d3d3d3",  # Non-significant
    1: "#e31a1c",  # High-High
    2: "#1f78b4",  # Low-Low
    3: "#a6cee3",  # Low-High
    4: "#fb9a99",  # High-Low
}
LISA_CMAP = ListedColormap([LISA_COLORS[i] for i in range(5)])

# Mode-weight bars use a palette distinct from the LISA red/blue (Brewer PuOr,
# colour-blind safe) so loading sign is not confused with cluster type.
POS_WEIGHT_COLOR = "#e66101"
NEG_WEIGHT_COLOR = "#5e3c99"


def lisa_legend_handles(ids=(1, 2, 3, 4, 0)) -> list[Patch]:
    """Legend patches for the given LISA ids."""
    return [Patch(color=LISA_COLORS[i], label=LISA_NAMES[i]) for i in ids]
